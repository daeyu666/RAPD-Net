"""Train the basis-invariant tangent-projected proposal Stage-2 variant."""
from __future__ import annotations

import argparse, json, math, os
from typing import Dict, List
import torch
import torch.nn.functional as F

from config import parse_args
from data_loader import build_loaders
from losses import SAMLoss
from metrics import MetricAverager, calc_metrics
from models.stage2_tangent_projected_proposal import Stage2TangentProjectedProposalNet
from train_stage2_coefficients import FixedSpatialDegradation, build_spectral_response, load_stage1_basis_checkpoint
from train_stage2_null_tangent_manifold import first_spectral_difference, second_spectral_difference, project, OnlinePearson
from utils import AverageMeter, CSVLogger, count_parameters, ensure_dir, get_device, move_to_device, save_checkpoint, set_seed, write_log


def _has_option(arguments: List[str], option: str) -> bool:
    return any(x == option or x.startswith(option + "=") for x in arguments)


def parse_specific_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--stage1_basis_checkpoint", type=str, default="./checkpoints/stage1_basis/PaviaU/basis_for_stage2.pth")
    p.add_argument("--anchor_ridge_ratio", type=float, default=1e-3)
    p.add_argument("--projector_tolerance", type=float, default=1e-6)
    p.add_argument("--tangent_dimension", type=int, default=4)
    p.add_argument("--tangent_kernel_size", type=int, default=5)
    p.add_argument("--tangent_dilation", type=int, default=2)
    p.add_argument("--tangent_chunk_pixels", type=int, default=2048)
    p.add_argument("--proposal_amplitude_multiplier", type=float, default=8.0)
    p.add_argument("--proposal_predictor_hidden", type=int, default=96)
    p.add_argument("--proposal_predictor_blocks", type=int, default=4)
    p.add_argument("--proposal_grad_clip", type=float, default=1.0)
    p.add_argument("--proposal_diagnose_only", action="store_true")
    p.add_argument("--proposal_lambda_l1", type=float, default=1.0)
    p.add_argument("--proposal_lambda_sam", type=float, default=0.3)
    p.add_argument("--proposal_lambda_sgrad1", type=float, default=0.1)
    p.add_argument("--proposal_lambda_sgrad2", type=float, default=0.05)
    p.add_argument("--proposal_lambda_residual", type=float, default=0.8)
    p.add_argument("--proposal_lambda_lr_hsi", type=float, default=0.2)
    p.add_argument("--proposal_lambda_lr_null", type=float, default=0.1)
    p.add_argument("--proposal_lambda_off_tangent", type=float, default=0.0)
    specific, remaining = p.parse_known_args()
    cfg = parse_args(remaining)
    for k, v in vars(specific).items(): setattr(cfg, k, v)
    if not _has_option(remaining, "--msi_mode"): cfg.msi_mode = "srf"
    if not _has_option(remaining, "--srf_band_set"): cfg.srf_band_set = "wv2_visible6"
    default = "./checkpoints/stage1_basis/PaviaU/basis_for_stage2.pth"
    if cfg.stage1_basis_checkpoint == default and cfg.dataset != "PaviaU":
        cfg.stage1_basis_checkpoint = os.path.join(cfg.checkpoint_root, "stage1_basis", cfg.dataset, "basis_for_stage2.pth")
    if cfg.tangent_dimension < 1: raise ValueError("tangent_dimension must be positive")
    if cfg.tangent_kernel_size < 3 or cfg.tangent_kernel_size % 2 == 0: raise ValueError("tangent_kernel_size must be odd >=3")
    if cfg.tangent_dilation < 1 or cfg.tangent_chunk_pixels < 1: raise ValueError("invalid tangent geometry")
    if cfg.proposal_amplitude_multiplier <= 0: raise ValueError("proposal_amplitude_multiplier must be positive")
    return cfg


@torch.no_grad()
def build_targets(model, out: Dict[str, torch.Tensor], gt: torch.Tensor):
    c_gt = model.stage1.encode(gt, basis=out["basis"])
    c_null = project(model.exact_null_projector.to(c_gt), c_gt)
    missing = c_null - out["null_seed_coefficients"]
    t = out["tangent_basis"]
    a = torch.einsum("nrdhw,nrhw->ndhw", t, missing)
    target = torch.einsum("nrdhw,ndhw->nrhw", t, a)
    target = project(model.exact_null_projector.to(target), target)
    return {"coeff": c_gt, "null": c_null, "missing": missing, "tangent": target}


def compute_losses(model, out, batch, hsi_deg, coeff_deg, sam_loss, cfg):
    gt, lr = batch["gt"], batch["lr_hsi"]
    pred = out["reconstructed_hsi"]
    hsi_l1 = F.l1_loss(pred, gt)
    sam = sam_loss(pred, gt)
    sg1 = F.l1_loss(first_spectral_difference(pred), first_spectral_difference(gt))
    sg2 = F.l1_loss(second_spectral_difference(pred), second_spectral_difference(gt))
    tar = build_targets(model, out, gt)
    scale = out["coefficient_scale"].view(1, -1, 1, 1)
    residual = F.smooth_l1_loss(out["tangent_residual"] / scale, tar["tangent"] / scale, beta=0.25)
    off = (out["off_tangent_proposal"] / scale).square().mean()
    lr_hsi = F.l1_loss(hsi_deg(pred, target_size=lr.shape[-2:]), lr)
    corrected_null = out["null_seed_coefficients"] + out["tangent_residual"]
    lr_null_target = project(model.exact_null_projector.to(out["lr_coefficients"]), out["lr_coefficients"])
    lr_null = F.smooth_l1_loss(coeff_deg(corrected_null, target_size=lr_null_target.shape[-2:]) / scale, lr_null_target / scale, beta=0.25)
    total = (cfg.proposal_lambda_l1*hsi_l1 + cfg.proposal_lambda_sam*sam + cfg.proposal_lambda_sgrad1*sg1 +
             cfg.proposal_lambda_sgrad2*sg2 + cfg.proposal_lambda_residual*residual + cfg.proposal_lambda_lr_hsi*lr_hsi +
             cfg.proposal_lambda_lr_null*lr_null + cfg.proposal_lambda_off_tangent*off)
    return {"total": total, "residual": residual, "off": off}, tar


def diagnostics(out):
    return {
        "rho_tan": float(out["tangent_projection_energy_ratio"].detach().item()),
        "rho_off": float(out["off_tangent_energy_ratio"].detach().item()),
        "sat": float(out["proposal_saturation_ratio"].detach().item()),
        "proposal_abs": float(out["global_coefficient_proposal"].detach().abs().mean().item()),
        "tangent_abs": float(out["tangent_residual"].detach().abs().mean().item()),
    }


def train_epoch(model, loader, opt, hsi_deg, coeff_deg, sam_loss, cfg, device):
    model.train(); model.stage1.eval()
    meters = {k: AverageMeter() for k in ["total", "residual", "off", "rho_tan", "rho_off", "sat"]}
    for batch in loader:
        batch = move_to_device(batch, device); opt.zero_grad(set_to_none=True)
        out = model(batch["lr_hsi"], batch["hr_msi"])
        losses, _ = compute_losses(model, out, batch, hsi_deg, coeff_deg, sam_loss, cfg)
        losses["total"].backward()
        if cfg.proposal_grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.proposal_predictor.parameters(), cfg.proposal_grad_clip)
        opt.step(); n = batch["lr_hsi"].size(0); d = diagnostics(out)
        for k in ["total", "residual", "off"]: meters[k].update(float(losses[k].detach().item()), n)
        for k in ["rho_tan", "rho_off", "sat"]: meters[k].update(d[k], n)
    return {k: v.avg for k, v in meters.items()}


@torch.no_grad()
def evaluate(model, loader, hsi_deg, coeff_deg, sam_loss, cfg, device):
    model.eval()
    metrics = {k: MetricAverager() for k in ["proposal", "anchor", "oracle", "basis"]}
    dm = {k: AverageMeter() for k in ["rho_tan", "rho_off", "sat", "proposal_abs", "tangent_abs"]}
    missing_e = pred_e = oracle_e = 0.0; corr = OnlinePearson(); leakage = 0.0
    for batch in loader:
        batch = move_to_device(batch, device); out = model(batch["lr_hsi"], batch["hr_msi"])
        _, tar = compute_losses(model, out, batch, hsi_deg, coeff_deg, sam_loss, cfg)
        oracle_hsi = model.stage1.decode(out["anchor_coefficients"] + tar["tangent"], basis=out["basis"])
        basis_hsi = model.stage1.decode(tar["coeff"], basis=out["basis"])
        for name, x in {"proposal": out["reconstructed_hsi"], "anchor": out["anchor_hsi"], "oracle": oracle_hsi, "basis": basis_hsi}.items():
            metrics[name].update(calc_metrics(x, batch["gt"], cfg.scale_ratio))
        n = batch["lr_hsi"].size(0)
        for k, v in diagnostics(out).items(): dm[k].update(v, n)
        miss = tar["missing"].double(); rem = (tar["missing"] - out["tangent_residual"]).double(); orem = (tar["missing"] - tar["tangent"]).double()
        missing_e += float(miss.square().sum().item()); pred_e += float(rem.square().sum().item()); oracle_e += float(orem.square().sum().item())
        corr.update(out["null_seed_coefficients"] + out["tangent_residual"], tar["null"])
        null_msi = torch.einsum("mr,nrhw->nmhw", model.reduced_response.to(out["tangent_residual"]), out["tangent_residual"])
        leakage = max(leakage, float(null_msi.abs().max().item()))
    r = {}
    for prefix, meter in metrics.items():
        for k, v in meter.average().items(): r[f"{prefix}_{k.lower()}"] = v
    for k, m in dm.items(): r[k] = m.avg
    missing_e = max(missing_e, 1e-30)
    r["capture"] = 1.0 - pred_e/missing_e; r["oracle_capture"] = 1.0 - oracle_e/missing_e
    r["null_rrmse"] = math.sqrt(pred_e/missing_e); r["null_pearson"] = corr.value(); r["null_leakage"] = leakage
    r["gain_psnr"] = r["proposal_psnr"] - r["anchor_psnr"]; r["gain_sam"] = r["anchor_sam"] - r["proposal_sam"]
    r["oracle_gap"] = r["oracle_psnr"] - r["proposal_psnr"]; r["observable_rank"] = int(model.observable_rank.item())
    return r


def main():
    cfg = parse_specific_args(); set_seed(cfg.seed); device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)
    stage1, _ = load_stage1_basis_checkpoint(cfg.stage1_basis_checkpoint, expected_n_bands=info["n_bands"], device=device)
    model = Stage2TangentProjectedProposalNet(stage1_model=stage1, spectral_response=build_spectral_response(info).to(device),
        anchor_ridge_ratio=cfg.anchor_ridge_ratio, projector_tolerance=cfg.projector_tolerance, tangent_dimension=cfg.tangent_dimension,
        tangent_kernel_size=cfg.tangent_kernel_size, tangent_dilation=cfg.tangent_dilation, tangent_chunk_pixels=cfg.tangent_chunk_pixels,
        proposal_amplitude_multiplier=cfg.proposal_amplitude_multiplier, predictor_hidden_channels=cfg.proposal_predictor_hidden,
        predictor_blocks=cfg.proposal_predictor_blocks).to(device)
    opt = torch.optim.AdamW(model.proposal_predictor.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs,1), eta_min=cfg.lr*0.05)
    hsi_deg = FixedSpatialDegradation(channels=info["n_bands"], kernel_size=5, sigma=2.0).to(device)
    coeff_deg = FixedSpatialDegradation(channels=stage1.basis_rank, kernel_size=5, sigma=2.0).to(device); sam_loss = SAMLoss()
    root = "stage2_tangent_projected_proposal"
    ckpt_dir = os.path.join(cfg.checkpoint_root, root, cfg.dataset); out_dir = os.path.join(cfg.output_root, root, cfg.dataset)
    log = os.path.join(cfg.log_root, root, f"{cfg.dataset}.log"); csv = os.path.join(cfg.log_root, root, f"{cfg.dataset}.csv")
    ensure_dir(ckpt_dir); ensure_dir(out_dir)
    fields = ["epoch","lr","train_total","proposal_psnr","proposal_sam","anchor_psnr","anchor_sam","oracle_psnr","basis_psnr","capture","oracle_capture","null_rrmse","null_pearson","rho_tan","rho_off","sat","oracle_gap"]
    logger = CSVLogger(csv, fields)
    start = evaluate(model, test_loader, hsi_deg, coeff_deg, sam_loss, cfg, device)
    write_log(log, f"Projected-proposal start | PSNR={start['proposal_psnr']:.4f} SAM={start['proposal_sam']:.4f} | anchor={start['anchor_psnr']:.4f}/{start['anchor_sam']:.4f} | oracle={start['oracle_psnr']:.4f} basis={start['basis_psnr']:.4f} | rho_tan={start['rho_tan']:.4f}")
    write_log(log, f"Model | params={count_parameters(model.proposal_predictor):.3f}M | d={cfg.tangent_dimension} | geometry={cfg.tangent_kernel_size}x{cfg.tangent_kernel_size} dilation={cfg.tangent_dilation} | amplitude={cfg.proposal_amplitude_multiplier:.2f}")
    if cfg.proposal_diagnose_only:
        with open(os.path.join(out_dir,"diagnose_only.json"),"w",encoding="utf-8") as f: json.dump(start,f,indent=2,ensure_ascii=False)
        return
    best, best_epoch = start["proposal_psnr"], 0; best_path = os.path.join(ckpt_dir,"projected_proposal_best_psnr.pth")
    save_checkpoint(model,opt,0,best,best_path,extra={"dataset":cfg.dataset,"tangent_dimension":cfg.tangent_dimension})
    for epoch in range(1,cfg.epochs+1):
        tr = train_epoch(model, train_loader, opt, hsi_deg, coeff_deg, sam_loss, cfg, device)
        ev = evaluate(model, test_loader, hsi_deg, coeff_deg, sam_loss, cfg, device); lr = opt.param_groups[0]["lr"]; sched.step()
        row = {"epoch":epoch,"lr":lr,"train_total":tr["total"], **{k:ev.get(k,"") for k in fields}}; row["epoch"]=epoch; row["lr"]=lr; row["train_total"]=tr["total"]; logger.write(row)
        write_log(log, f"Epoch {epoch:03d}/{cfg.epochs:03d} | PSNR={ev['proposal_psnr']:.4f} SAM={ev['proposal_sam']:.4f} | gain={ev['gain_psnr']:+.4f} dB/{ev['gain_sam']:+.4f} deg | capture={100*ev['capture']:.2f}% | rho_tan={ev['rho_tan']:.4f} off={ev['rho_off']:.4f} sat={100*ev['sat']:.2f}% | null={ev['null_rrmse']:.4f}, r={ev['null_pearson']:.4f}")
        if ev["proposal_psnr"] > best:
            best, best_epoch = ev["proposal_psnr"], epoch; save_checkpoint(model,opt,epoch,best,best_path,extra={"dataset":cfg.dataset,"tangent_dimension":cfg.tangent_dimension})
    summary = {"dataset":cfg.dataset,"best_epoch":best_epoch,"best_psnr":best,"checkpoint":best_path,"start":start,"tangent_dimension":cfg.tangent_dimension}
    with open(os.path.join(out_dir,"training_summary.json"),"w",encoding="utf-8") as f: json.dump(summary,f,indent=2,ensure_ascii=False)
    print(f"Best projected-proposal PSNR={best:.4f} at epoch {best_epoch}")


if __name__ == "__main__": main()
