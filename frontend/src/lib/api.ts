/** API client for the FastAPI backend (proxied through Next.js rewrites). */

const BASE = "";

export async function fetchAPI<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options?.headers },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "Unknown error");
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

export const api = {
  // Generation
  generate: (params: Record<string, unknown>) =>
    fetchAPI("/api/generate", { method: "POST", body: JSON.stringify(params) }),
  getGenerationHistory: (page = 1, pageSize = 50) =>
    fetchAPI(`/api/generate/history?page=${page}&page_size=${pageSize}`),

  // Training
  getTrainingStatus: () => fetchAPI("/api/training/status"),
  startTraining: (config: Record<string, unknown>) =>
    fetchAPI("/api/training/start", { method: "POST", body: JSON.stringify(config) }),
  stopTraining: () => fetchAPI("/api/training/stop", { method: "POST" }),
  getTrainingHistory: () => fetchAPI("/api/training/history"),
  getRunMetrics: (runId: number) => fetchAPI(`/api/training/runs/${runId}/metrics`),
  getRunSamples: (runId: number) => fetchAPI(`/api/training/runs/${runId}/samples`),
  getCheckpoints: (runId?: number) =>
    fetchAPI(`/api/training/checkpoints${runId ? `?run_id=${runId}` : ""}`),

  // Dataset
  getDatasetStats: () => fetchAPI("/api/dataset/stats"),
  getDatasetImages: (page = 1, pageSize = 20) =>
    fetchAPI(`/api/dataset/images?page=${page}&page_size=${pageSize}`),
  uploadImage: async (file: File) => {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/api/dataset/upload", { method: "POST", body: form });
    if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
    return res.json();
  },

  // Dedup & Cleanup
  deduplicate: () => fetchAPI("/api/dataset/deduplicate", { method: "POST", body: JSON.stringify({}) }),
  cleanupSquareImages: () => fetchAPI("/api/dataset/cleanup", { method: "POST" }),

  // Scraper
  getScraperStatus: () => fetchAPI("/api/scraper/status"),
  startScraper: (count: number, scrapeType: string) =>
    fetchAPI("/api/scraper/start", { method: "POST", body: JSON.stringify({ count, scrape_type: scrapeType }) }),
  stopScraper: () => fetchAPI("/api/scraper/stop", { method: "POST" }),

  // Settings
  getStorage: () => fetchAPI("/api/settings/storage"),
  archive: (target: string) =>
    fetchAPI("/api/settings/archive", { method: "POST", body: JSON.stringify({ target }) }),
  deleteData: (target: string) =>
    fetchAPI("/api/settings/delete", { method: "POST", body: JSON.stringify({ target, confirm: true }) }),
  deleteArchives: () =>
    fetchAPI("/api/settings/delete-archives", { method: "POST" }),

  // Health
  health: () => fetchAPI("/health"),
};
