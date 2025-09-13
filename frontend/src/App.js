import React, { useState, useEffect } from "react";
import './App.css';

function App() {
  const [running, setRunning] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [summary, setSummary] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [selectedSession, setSelectedSession] = useState("");

  // ambil daftar sessions setiap kali selesai tracking
  useEffect(() => {
    fetch("http://localhost:5000/sessions")
      .then((res) => res.json())
      .then((data) => setSessions(data))
      .catch((err) => console.error("Error fetching sessions:", err));
  }, [running, summary]);

  const handleStart = async () => {
    try {
      const res = await fetch("http://localhost:5000/start_session", {
        method: "POST",
      });
      const data = await res.json();
      setSessionId(data.session_id);
      setRunning(true);
      setSummary(null);
    } catch (err) {
      console.error("Error starting session:", err);
    }
  };

  const handleStop = async () => {
    try {
      await fetch("http://localhost:5000/stop_session", { method: "POST" });

      const res = await fetch(
        `http://localhost:5000/summary?session_id=${sessionId}`
      );
      const data = await res.json();
      setSummary(data);
      setRunning(false);
    } catch (err) {
      console.error("Error stopping session:", err);
    }
  };

  const handleViewSummary = async () => {
    if (!selectedSession) return;
    try {
      const res = await fetch(
        `http://localhost:5000/summary?session_id=${selectedSession}`
      );
      const data = await res.json();
      setSummary(data);
      setSessionId(selectedSession);
    } catch (err) {
      console.error("Error fetching summary:", err);
    }
  };

  return (
    <div className="container">
      <div className="header">
        <h1 className="title">Emotion Tracking</h1>
      </div>

      <div className="main-card">
        {!running && (
          <div className="button-container">
            <button
              onClick={handleStart}
              className="button start-button"
            >
              Start Tracking
            </button>
          </div>
        )}

        {running && (
          <div className="video-section">
            <div className="video-container">
              <img
                src="http://localhost:5000/video_feed"
                alt="Live Emotion Feed"
                className="video"
              />
            </div>
            <button
              onClick={handleStop}
              className="button stop-button"
            >
              Stop & Show Summary
            </button>
          </div>
        )}

        {/* Session Selector */}
        {!running && (
          <div className="session-selector">
            <h3 className="section-title">View Previous Sessions</h3>
            <div className="select-container">
              <select
                value={selectedSession}
                onChange={(e) => setSelectedSession(e.target.value)}
                className="select"
              >
                <option value="">-- Select Session --</option>
                {sessions.map((s) => (
                  <option key={s.id} value={s.id}>
                    Session {s.id} ({new Date(s.start_time).toLocaleString()})
                  </option>
                ))}
              </select>
              <button
                onClick={handleViewSummary}
                className="button view-button"
              >
                View Summary
              </button>
            </div>
          </div>
        )}

        {/* Summary */}
        {summary && (
          <div className="summary-container">
            <div className="summary-header">
              <h2 className="summary-title">Session {sessionId} Summary</h2>
            </div>
            
            {Object.keys(summary).length === 0 ? (
              <div className="no-data">
                No emotion data recorded for this session
              </div>
            ) : (
              <div className="persons-grid">
                {Object.entries(summary).map(([person, data]) => (
                  <div key={person} className="person-card">
                    <h3 className="person-name">{person}</h3>
                    <p className="total-time">Total Time: {data.total_time.toFixed(1)} seconds</p>
                    <table className="table">
                      <thead className="table-header">
                        <tr>
                          <th className="table-header-cell">Emotion</th>
                          <th className="table-header-cell">Duration</th>
                          <th className="table-header-cell">Percentage</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(data.emotions).map(([emo, dur], index) => (
                          <tr key={emo} className={index % 2 === 0 ? "table-row" : "table-row-alt"}>
                            <td className="table-cell emotion-name">{emo}</td>
                            <td className="table-cell">{dur.toFixed(1)}s</td>
                            <td className="table-cell">{data.percentages[emo].toFixed(1)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;