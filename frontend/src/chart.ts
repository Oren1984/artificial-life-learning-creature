// Fitness-per-generation line chart (one y-axis, two series, hover crosshair).
import { chartSeries, niceTicks, scale } from "./logic";
import type { GenerationRecord } from "./types";

export const SERIES = [
  { key: "best", label: "Best creature", color: "#3987e5" },
  { key: "mean", label: "Population average", color: "#d95926" },
] as const;

const PAD = { left: 44, right: 16, top: 12, bottom: 30 };

export class FitnessChart {
  private data: GenerationRecord[] = [];
  private total = 1;
  private hover: number | null = null;
  private marked: number | null = null;

  constructor(
    private canvas: HTMLCanvasElement,
    private tooltip: HTMLElement,
    onPick: (generation: number) => void,
  ) {
    canvas.addEventListener("pointermove", (e) => {
      this.hover = this.indexAt(e);
      this.draw();
    });
    canvas.addEventListener("pointerleave", () => {
      this.hover = null;
      this.draw();
    });
    canvas.addEventListener("click", (e) => {
      const i = this.indexAt(e);
      if (i !== null) onPick(this.data[i].generation);
    });
  }

  update(data: GenerationRecord[], totalGenerations: number, marked: number | null): void {
    this.data = data;
    this.total = Math.max(1, totalGenerations);
    this.marked = marked;
    this.draw();
  }

  private indexAt(e: PointerEvent | MouseEvent): number | null {
    if (!this.data.length) return null;
    const rect = this.canvas.getBoundingClientRect();
    const plotW = rect.width - PAD.left - PAD.right;
    const g = Math.round(((e.clientX - rect.left - PAD.left) / plotW) * this.total);
    return Math.max(0, Math.min(this.data.length - 1, g));
  }

  draw(): void {
    const c = this.canvas;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const w = c.clientWidth || 600;
    const h = c.clientHeight || 260;
    const dpr = window.devicePixelRatio || 1;
    if (c.width !== Math.round(w * dpr)) {
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const s = chartSeries(this.data);
    const values = [...s.best, ...s.mean, 0];
    const ticks = niceTicks(Math.min(...values), Math.max(...values, 10));
    const sx = scale([0, this.total], [PAD.left, w - PAD.right]);
    const sy = scale([ticks[0], ticks[ticks.length - 1]], [h - PAD.bottom, PAD.top]);

    ctx.font = "11px system-ui, sans-serif";
    ctx.fillStyle = "#8f9bb3";
    ctx.strokeStyle = "rgba(255,255,255,0.07)";
    ctx.lineWidth = 1;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const tk of ticks) {
      ctx.beginPath();
      ctx.moveTo(PAD.left, sy(tk));
      ctx.lineTo(w - PAD.right, sy(tk));
      ctx.stroke();
      ctx.fillText(String(tk), PAD.left - 8, sy(tk));
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const tk of niceTicks(0, this.total, 5).filter((v) => v <= this.total)) {
      ctx.fillText(String(tk), sx(tk), h - PAD.bottom + 8);
    }

    if (this.marked !== null && this.marked < this.data.length) {
      ctx.strokeStyle = "rgba(255,255,255,0.35)";
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(sx(this.marked), PAD.top);
      ctx.lineTo(sx(this.marked), h - PAD.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    for (const series of SERIES) {
      const ys = s[series.key];
      ctx.strokeStyle = series.color;
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.beginPath();
      ys.forEach((v, i) => (i ? ctx.lineTo(sx(s.x[i]), sy(v)) : ctx.moveTo(sx(s.x[i]), sy(v))));
      ctx.stroke();
      if (ys.length) {
        const i = this.hover ?? ys.length - 1;
        ctx.fillStyle = series.color;
        ctx.strokeStyle = "#0d1220";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(sx(s.x[i]), sy(ys[i]), 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    }

    if (this.hover !== null && this.data[this.hover]) {
      const g = this.data[this.hover];
      const x = sx(g.generation);
      ctx.strokeStyle = "rgba(255,255,255,0.25)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, PAD.top);
      ctx.lineTo(x, h - PAD.bottom);
      ctx.stroke();
      this.tooltip.hidden = false;
      this.tooltip.textContent =
        `Generation ${g.generation} · best ${g.best_fitness.toFixed(1)} · ` +
        `average ${g.mean_fitness.toFixed(1)} · best food ${g.best_food.toFixed(2)}/6`;
      this.tooltip.style.left = `${Math.min(Math.max(x, 130), w - 130)}px`;
    } else {
      this.tooltip.hidden = true;
    }
  }
}
