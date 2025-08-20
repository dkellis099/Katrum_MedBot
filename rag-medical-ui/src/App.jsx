import { useMemo, useRef, useState } from "react";

// Optional logo (direct URL). If you don't want an env, just hardcode a URL or leave empty.
const LOGO_URL = import.meta.env.VITE_LOGO_URL || "";
const API_URL = "https://katrum-medbot.onrender.com";

/**
 * This version uses RELATIVE API paths:
 *   - /api/ask
 *   - /healthz
 * In dev: configure Vite proxy in vite.config.js
 * In prod (Vercel): add rewrites in vercel.json to your backend.
 */
export default function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [cites, setCites] = useState([]);
  const [elapsed, setElapsed] = useState(null);
  const [chatModel, setChatModel] = useState("llama2");
  const [embedModel, setEmbedModel] = useState("nomic-embed-text");
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState("");
  const esRef = useRef(null);

  const metrics = useMemo(
    () => [
      ["Status", loading ? "Streaming…" : "Idle"],
      ["Elapsed", elapsed !== null ? `${elapsed}s` : "—"],
      ["Chat Model", chatModel],
      ["Embed Model", embedModel],
    ],
    [loading, elapsed, chatModel, embedModel]
  );

  const ask = () => {
    setErrors("");
    if (!question.trim() || loading) return;

    setLoading(true);
    setAnswer("");
    setCites([]);
    setElapsed(null);

    // Build a SAME-ORIGIN URL to /api/ask
    const url = new URL("/api/ask", window.location.origin);
    url.searchParams.set("question", question);
    url.searchParams.set("chat_model", chatModel);
    url.searchParams.set("embed_model", embedModel);

    const es = new EventSource(url.toString());
    esRef.current = es;

    // unnamed "message" event (some servers send first chunk here)
    es.onmessage = (e) => {
      try {
        setAnswer((v) => v + JSON.parse(e.data));
      } catch {
        setAnswer((v) => v + String(e.data));
      }
    };

    es.addEventListener("token", (e) => {
      try {
        setAnswer((v) => v + JSON.parse(e.data));
      } catch {
        setAnswer((v) => v + String(e.data));
      }
    });

    es.addEventListener("done", (e) => {
      try {
        const m = JSON.parse(e.data);
        setCites(m.cites || []);
        setElapsed(m.elapsed ?? null);
      } catch {}
      setLoading(false);
      es.close();
      esRef.current = null;

      // tidy punctuation/spacing
      setAnswer((v) =>
        v.replace(/\s+([.,;:!?])/g, "$1").replace(/\s{2,}/g, " ").trim()
      );
    });

    es.onerror = () => {
      setLoading(false);
      es.close();
      esRef.current = null;
      setErrors("Connection error. Check API and rewrites/proxy.");
    };
  };

  const clearAll = () => {
    setQuestion("");
    setAnswer("");
    setCites([]);
    setElapsed(null);
    setErrors("");
  };

  return (
    <div className="gtr-root">
      {/* Top navbar */}
      <nav className="navbar">
        <div className="container nav-inner">
          <div className="brand">
            {LOGO_URL ? (
              <img
                src={LOGO_URL}
                alt="Katrum"
                className="brand-logo"
                onError={(e) => {
                  e.currentTarget.style.display = "none";
                }}
              />
            ) : null}
            <span className="brand-name">Katrum Medical Assistant</span>
          </div>
        </div>
      </nav>

      {/* Page body */}
      <div className="container">
        <div className="row">
          {/* LEFT rail */}
          <aside className="col-left">
            <div className="panel">
              <div className="panel-title">Overview</div>
              <table className="table zebra">
                <tbody>
                  {metrics.map(([k, v]) => (
                    <tr key={String(k)}>
                      <td className="key">{k}</td>
                      <td className="val">{String(v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="panel">
              <div className="panel-title">Settings</div>
              <div className="form">
                <label>
                  <span>Chat Model</span>
                  <input
                    value={chatModel}
                    onChange={(e) => setChatModel(e.target.value)}
                  />
                </label>
                <label>
                  <span>Embed Model</span>
                  <input
                    value={embedModel}
                    onChange={(e) => setEmbedModel(e.target.value)}
                  />
                </label>

                {/* Relative health check */}
                <a className="btn-ghost" href="/healthz" target="_blank" rel="noreferrer">
                  Health check
                </a>
              </div>
            </div>
          </aside>

          {/* MAIN content */}
          <main className="col-main">
            <h1 className="page-title">Ask</h1>

            <div className="panel" style={{ marginInline: "auto", maxWidth: 720 }}>
              <div className="panel-body">
                <label className="sr-only">Question</label>
                <textarea
                  className="ask-input"
                  placeholder="e.g., What are the primary functions of RNA?"
                  rows={5}
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                />
                {errors && <div className="error">{errors}</div>}
                <div className="actions" style={{ justifyContent: "center" }}>
                  <button className="btn-primary" disabled={loading} onClick={ask}>
                    {loading ? "Streaming…" : "Ask"}
                  </button>
                  <button className="btn-light" onClick={clearAll}>
                    Clear
                  </button>
                </div>
              </div>
            </div>

            <h2 className="section-title" style={{ textAlign: "center" }}>
              Answer
            </h2>
            <div className="panel" style={{ marginInline: "auto", maxWidth: 720 }}>
              <div className="panel-body">
                <div className={`answer ${loading ? "loading" : ""}`}>
                  {answer || <span className="muted">No answer yet.</span>}
                </div>
                {!!cites.length && (
                  <div className="chips" style={{ justifyContent: "center" }}>
                    {cites.map((c) => (
                      <span key={c} className="chip">
                        {c}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* credit */}
            <div style={{ textAlign: "center", marginTop: 24, color: "#555" }}>
              Built by <strong>Dylan Ellis</strong>
            </div>
          </main>
        </div>
      </div>

      {/* Footer */}
      <footer className="footer">
        <div className="container footer-inner">
          <div>© {new Date().getFullYear()} Katrum</div>
          <div className="muted">
            Answers are for information only — not medical advice.
          </div>
        </div>
      </footer>
    </div>
  );
}


