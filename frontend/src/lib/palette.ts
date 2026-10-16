import { useEffect, useState } from "react";
import type { ModelName } from "./types";

/** Chart colours read from the CSS tokens, so charts follow light / dark mode. */
export interface Palette {
  series: string[];
  ink: string;
  ink2: string;
  muted: string;
  grid: string;
  axis: string;
  surface: string;
}

function read(): Palette {
  const css = getComputedStyle(document.documentElement);
  const v = (name: string, fallback: string) => css.getPropertyValue(name).trim() || fallback;
  return {
    series: [1, 2, 3, 4, 5, 6].map((i) =>
      v(`--series-${i}`, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"][i - 1]),
    ),
    ink: v("--ink", "#0b0b0b"),
    ink2: v("--ink-2", "#52514e"),
    muted: v("--muted", "#6f6d68"),
    grid: v("--grid", "#e1e0d9"),
    axis: v("--axis", "#c3c2b7"),
    surface: v("--surface", "#fcfcfb"),
  };
}

export function usePalette(): Palette {
  const [palette, setPalette] = useState<Palette>(read);
  useEffect(() => {
    const media = window.matchMedia?.("(prefers-color-scheme: dark)");
    const update = () => setPalette(read());
    media?.addEventListener("change", update);
    return () => media?.removeEventListener("change", update);
  }, []);
  return palette;
}

/** Fixed colour + dash per model, so identity never relies on colour alone. */
export const MODEL_STYLE: Record<ModelName, { slot: number; dash?: string }> = {
  logistic_regression: { slot: 0 },
  mlp: { slot: 1, dash: "6 4" },
  lstm: { slot: 2, dash: "2 3" },
};
