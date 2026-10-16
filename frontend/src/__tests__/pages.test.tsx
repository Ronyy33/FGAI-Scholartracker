import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BandChip } from "../components/ui";
import { OverviewPage } from "../pages/OverviewPage";

const overview = {
  synthetic_data: true, records_processed: 8561, students: 1200, current_academic_year: "2026-27",
  current_students_scored: 381, flagged: 71,
  risk_bands: [
    { band: "Low", range: [0, 29], count: 310, percent: 81.4 },
    { band: "Moderate", range: [30, 59], count: 34, percent: 8.9 },
    { band: "High", range: [60, 100], count: 37, percent: 9.7 },
  ],
  model: { run_id: "run-1", default_model: "logistic_regression", test_roc_auc: 0.878, test_pr_auc: 0.705, test_positive_rate: 0.204 },
};

afterEach(() => vi.unstubAllGlobals());

describe("BandChip", () => {
  it("always shows the band as text, not colour alone", () => {
    render(<BandChip band="High" />);
    expect(screen.getByText("High")).toBeInTheDocument();
    expect(screen.getByTitle("High risk band")).toBeInTheDocument();
  });
});

describe("OverviewPage", () => {
  it("renders figures from the API", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(overview), { status: 200 })));
    render(<MemoryRouter><OverviewPage /></MemoryRouter>);
    expect(await screen.findByText("8,561")).toBeInTheDocument();
    expect(screen.getByText("381")).toBeInTheDocument();
    expect(screen.getByText(/310 students · 81.4 %/)).toBeInTheDocument();
    expect(screen.getByText("0.878")).toBeInTheDocument();
  });

  it("shows an error state with the server message", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(
      JSON.stringify({ error: { code: "demo_unavailable", message: "Synthetic demo data is not available on this server." } }),
      { status: 503 })));
    render(<MemoryRouter><OverviewPage /></MemoryRouter>);
    expect(await screen.findByRole("alert")).toHaveTextContent("not available on this server");
  });
});
