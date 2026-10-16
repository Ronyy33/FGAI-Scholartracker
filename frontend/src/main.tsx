import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import "./index.css";
import { AboutPage } from "./pages/AboutPage";
import { AnalyticsPage } from "./pages/AnalyticsPage";
import { BatchPage } from "./pages/BatchPage";
import { EvaluationPage } from "./pages/EvaluationPage";
import { OverviewPage } from "./pages/OverviewPage";
import { StudentDetailPage, StudentsPage } from "./pages/StudentsPage";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<OverviewPage />} />
          <Route path="students" element={<StudentsPage />} />
          <Route path="students/:prn" element={<StudentDetailPage />} />
          <Route path="batch" element={<BatchPage />} />
          <Route path="analytics" element={<AnalyticsPage />} />
          <Route path="evaluation" element={<EvaluationPage />} />
          <Route path="about" element={<AboutPage />} />
          <Route path="*" element={<p className="text-ink-2">Page not found.</p>} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);
