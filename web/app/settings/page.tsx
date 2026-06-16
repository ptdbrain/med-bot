import { API_BASE_URL } from "@/lib/api/client";

export default function SettingsPage() {
  return (
    <main className="workspace-page workspace-page--narrow">
      <header className="page-header">
        <div>
          <div className="eyebrow">Workspace</div>
          <h1>Cài đặt</h1>
          <p>Cấu hình mặc định của giao diện MedTutor.</p>
        </div>
      </header>

      <section className="settings-section">
        <div className="settings-copy">
          <h2>Kết nối backend</h2>
          <p>Được cấu hình qua biến môi trường của frontend.</p>
        </div>
        <label className="field">
          <span>FastAPI base URL</span>
          <input value={API_BASE_URL} readOnly />
        </label>
      </section>

      <section className="settings-section">
        <div className="settings-copy">
          <h2>Mặc định RAG</h2>
          <p>Chat đang lưu lựa chọn mode và top K trong trình duyệt.</p>
        </div>
        <div className="settings-readonly">
          <span>Mode mặc định</span>
          <strong>Hybrid</strong>
          <span>Top K</span>
          <strong>5</strong>
        </div>
      </section>

      <aside className="safety-footnote">
        Secrets và API key chỉ được cấu hình ở backend, không lưu trong trình
        duyệt.
      </aside>
    </main>
  );
}
