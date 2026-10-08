// Canvas rendering of one world pane. Draws only what the replay data says.
import { frameAt } from "./logic";
import type { Replay, Scenario } from "./types";

export interface PaneStyle {
  body: string; // creature colour
  glow: string;
}

export const BASELINE_STYLE: PaneStyle = { body: "#9fb4d8", glow: "rgba(130,160,220,0.55)" };
export const TRAINED_STYLE: PaneStyle = { body: "#7df9e0", glow: "rgba(70,240,210,0.6)" };

const TRAIL_LENGTH = 90;

/** Size the canvas backing store to its CSS box; returns the side in CSS pixels. */
export function fitCanvas(canvas: HTMLCanvasElement): number {
  const side = canvas.clientWidth || 300;
  const dpr = window.devicePixelRatio || 1;
  const px = Math.round(side * dpr);
  if (canvas.width !== px) {
    canvas.width = px;
    canvas.height = px;
  }
  return side;
}

export function drawPane(
  canvas: HTMLCanvasElement,
  scenario: Scenario,
  replay: Replay,
  t: number,
  radii: { creature: number; food: number },
  style: PaneStyle,
  showTrail: boolean,
  now: number,
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const side = fitCanvas(canvas);
  const dpr = canvas.width / side;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const S = side;
  const frame = frameAt(replay, t);

  // Backdrop
  const bg = ctx.createRadialGradient(S * 0.5, S * 0.4, S * 0.05, S * 0.5, S * 0.5, S * 0.8);
  bg.addColorStop(0, "#131c2e");
  bg.addColorStop(1, "#070a12");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, S, S);
  ctx.strokeStyle = "rgba(255,255,255,0.035)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 1; i < 10; i++) {
    ctx.moveTo((i * S) / 10, 0);
    ctx.lineTo((i * S) / 10, S);
    ctx.moveTo(0, (i * S) / 10);
    ctx.lineTo(S, (i * S) / 10);
  }
  ctx.stroke();

  // Hazards: pulsing red fields with a dashed rim
  const pulse = 0.5 + 0.5 * Math.sin(now / 420);
  for (const [x, y, r] of scenario.hazards) {
    const g = ctx.createRadialGradient(x * S, y * S, 0, x * S, y * S, r * S);
    g.addColorStop(0, `rgba(255,70,90,${0.42 + 0.14 * pulse})`);
    g.addColorStop(1, "rgba(255,70,90,0.08)");
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(x * S, y * S, r * S, 0, Math.PI * 2);
    ctx.fill();
    ctx.setLineDash([4, 4]);
    ctx.lineDashOffset = -now / 80;
    ctx.strokeStyle = "rgba(255,110,125,0.85)";
    ctx.lineWidth = 1.2;
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Obstacles: solid rocks
  for (const [x, y, r] of scenario.obstacles) {
    const g = ctx.createRadialGradient((x - r * 0.3) * S, (y - r * 0.3) * S, 0, x * S, y * S, r * S);
    g.addColorStop(0, "#56617a");
    g.addColorStop(1, "#262d3d");
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(x * S, y * S, r * S, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#8793ad";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // Food: glowing particles; eaten ones leave a faint ring
  scenario.food.forEach(([x, y], i) => {
    const cx = x * S;
    const cy = y * S;
    if (!frame.foodLeft[i]) {
      ctx.strokeStyle = "rgba(190,255,120,0.22)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(cx, cy, radii.food * S, 0, Math.PI * 2);
      ctx.stroke();
      return;
    }
    const wobble = 1 + 0.18 * Math.sin(now / 260 + i * 1.7);
    ctx.shadowColor = "rgba(190,255,90,0.95)";
    ctx.shadowBlur = 14;
    ctx.fillStyle = "#d4ff7a";
    ctx.beginPath();
    ctx.arc(cx, cy, radii.food * S * 0.75 * wobble, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
  });

  // Trail
  if (showTrail) {
    const from = Math.max(0, frame.step - TRAIL_LENGTH);
    for (let i = from; i < frame.step; i++) {
      const a = (i - from) / TRAIL_LENGTH;
      ctx.strokeStyle = style.glow.replace(/[\d.]+\)$/, `${(a * 0.6).toFixed(3)})`);
      ctx.lineWidth = 1 + a * 3;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(replay.x[i] * S, replay.y[i] * S);
      ctx.lineTo(replay.x[i + 1] * S, replay.y[i + 1] * S);
      ctx.stroke();
    }
  }

  // Creature
  const dead = frame.ended && replay.summary.dead;
  const cx = frame.x * S;
  const cy = frame.y * S;
  const r = radii.creature * S;
  const body = dead ? "#5b6170" : frame.inHazard ? "#ff8a96" : style.body;
  ctx.shadowColor = dead ? "transparent" : frame.inHazard ? "rgba(255,80,100,0.9)" : style.glow;
  ctx.shadowBlur = 22;
  ctx.fillStyle = body;
  ctx.beginPath();
  ctx.arc(cx, cy, r * (1 + (dead ? 0 : 0.06 * Math.sin(now / 150))), 0, Math.PI * 2);
  ctx.fill();
  ctx.shadowBlur = 0;
  const next = Math.min(frame.step + 1, replay.x.length - 1);
  const dx = replay.x[next] - replay.x[frame.step];
  const dy = replay.y[next] - replay.y[frame.step];
  const len = Math.hypot(dx, dy) || 1;
  ctx.fillStyle = "#06101c";
  for (const sgn of [-1, 1]) {
    const ex = cx + (dx / len) * r * 0.45 + (-dy / len) * r * 0.38 * sgn;
    const ey = cy + (dy / len) * r * 0.45 + (dx / len) * r * 0.38 * sgn;
    ctx.beginPath();
    ctx.arc(ex, ey, Math.max(1, r * 0.2), 0, Math.PI * 2);
    ctx.fill();
  }
  if (frame.contact && !dead) {
    ctx.strokeStyle = "rgba(255,255,255,0.8)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(cx, cy, r * 1.7, 0, Math.PI * 2);
    ctx.stroke();
  }

  // Energy bar
  const bw = S * 0.3;
  ctx.fillStyle = "rgba(255,255,255,0.12)";
  ctx.fillRect(10, 10, bw, 5);
  ctx.fillStyle = frame.energy > 0.3 ? "#8be28b" : "#ff7a85";
  ctx.fillRect(10, 10, bw * Math.max(0, Math.min(1, frame.energy)), 5);
}
