import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Upload,
  Send,
  Plus,
  Sparkles,
  Activity,
  Brain,
  ShieldCheck,
  ChevronDown,
  BarChart3,
  FileSpreadsheet,
  Bot,
  User
} from 'lucide-react';

import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter
} from 'recharts';

import './styles.css';


// ============================================================
// API CONFIGURATION
// ============================================================

const API =
  import.meta.env.VITE_API_URL ||
  'http://localhost:5000';


// ============================================================
// API FETCH HELPER
// ============================================================

const apiFetch = async (path, options = {}) => {
  const controller = new AbortController();

  const timer = setTimeout(() => {
    controller.abort();
  }, 30000);

  try {
    const response = await fetch(API + path, {
      ...options,
      signal: controller.signal
    });

    let data = null;

    try {
      data = await response.json();
    } catch {
      data = null;
    }

    if (!response.ok) {
      throw new Error(
        data?.error ||
        data?.message ||
        `Backend request failed (${response.status}).`
      );
    }

    return data;

  } catch (error) {

    if (error.name === 'AbortError') {
      throw new Error(
        'The backend request timed out. Make sure Flask is running and retry.'
      );
    }

    if (error instanceof TypeError) {
      throw new Error(
        'Cannot reach the UDI backend. Start Flask on the configured API URL and retry.'
      );
    }

    throw error;

  } finally {
    clearTimeout(timer);
  }
};


// ============================================================
// FALLBACK DOMAINS
// ============================================================

const fallbackDomains = [
  {
    id: 'company',
    label: 'Company',
    icon: '▣',
    description:
      'Sales, demand, inventory, customers and profitability.'
  },
  {
    id: 'hospital',
    label: 'Hospital',
    icon: '✚',
    description:
      'Patient load, capacity, waiting time and staffing.'
  },
  {
    id: 'college',
    label: 'College',
    icon: '⌂',
    description:
      'Attendance, students, faculty and resource utilization.'
  },
  {
    id: 'municipality',
    label: 'Municipality',
    icon: '◈',
    description:
      'Waste, complaints, service demand and zones.'
  },
  {
    id: 'ngo',
    label: 'NGO',
    icon: '♡',
    description:
      'Beneficiaries, donations, aid and volunteer demand.'
  },
  {
    id: 'community',
    label: 'Community',
    icon: '⌁',
    description:
      'Local requests, participation, shortages and needs.'
  },
  {
    id: 'event_organizer',
    label: 'Event Organizer',
    icon: '★',
    description:
      'Attendance, capacity, registrations and resources.'
  }
];


// ============================================================
// MAIN APP
// ============================================================

function App() {

  const [domains, setDomains] =
    useState(fallbackDomains);

  const [domain, setDomain] =
    useState('company');

  const [session, setSession] =
    useState('');

  const [profile, setProfile] =
    useState(null);

  const [messages, setMessages] =
    useState([]);

  const [input, setInput] =
    useState('');

  const [loading, setLoading] =
    useState(false);

  const [uploading, setUploading] =
    useState(false);

  const [health, setHealth] =
    useState(null);

  const fileRef =
    useRef(null);


  // ==========================================================
  // INITIALIZATION
  // ==========================================================

  useEffect(() => {

    Promise.all([
      apiFetch('/api/domains'),
      apiFetch('/api/health')
    ])
      .then(([domainData, healthData]) => {

        if (Array.isArray(domainData)) {
          setDomains(domainData);
        }

        setHealth(healthData);

      })
      .catch(() => {

        setHealth({
          gemini_enabled: false,
          backend_offline: true
        });

      });


    apiFetch('/api/session', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        domain: 'company'
      })
    })
      .then((data) => {
        if (data?.session_id) {
          setSession(data.session_id);
        }
      })
      .catch(() => {});

  }, []);


  // ==========================================================
  // CURRENT DOMAIN
  // ==========================================================

  const current =
    domains.find((item) => item.id === domain) ||
    fallbackDomains[0];


  // ==========================================================
  // SELECT DOMAIN
  // ==========================================================

  const selectDomain = async (id) => {

    setDomain(id);
    setProfile(null);
    setMessages([]);

    if (session) {

      try {

        await apiFetch('/api/new-chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            session_id: session,
            domain: id
          })
        });

      } catch {
        // Domain switching should still work
        // even if the backend session reset fails.
      }

    }
  };


  // ==========================================================
  // UPLOAD DATASET
  // ==========================================================

  const upload = async (file) => {

    if (!file) {
      return;
    }

    setUploading(true);

    const formData = new FormData();

    formData.append('file', file);
    formData.append('domain', domain);
    formData.append('session_id', session);

    try {

      const data = await apiFetch('/api/upload', {
        method: 'POST',
        body: formData
      });

      setProfile(data.profile);

      setMessages([
        {
          role: 'assistant',
          content:
            `${file.name} is loaded. ` +
            `I profiled ${data.profile.rows.toLocaleString()} rows ` +
            `across ${data.profile.columns.length} columns. ` +
            `Ask me anything about the data.`
        }
      ]);

    } catch (error) {

      setMessages([
        {
          role: 'assistant',
          error: true,
          content: error.message
        }
      ]);

    } finally {

      setUploading(false);

    }
  };


  // ==========================================================
  // NEW CHAT
  // ==========================================================

  const newChat = async () => {

    try {

      const data = await apiFetch('/api/new-chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          session_id: session,
          domain
        })
      });

      if (data?.session_id) {
        setSession(data.session_id);
      }

      setMessages([]);
      setProfile(null);
      setInput('');

    } catch (error) {

      setMessages([
        {
          role: 'assistant',
          error: true,
          content: error.message
        }
      ]);

    }
  };


  // ==========================================================
  // ASK QUESTION
  // ==========================================================

  const ask = async (question) => {

    const query =
      (question || input || '').trim();

    if (!query || loading) {
      return;
    }

    setInput('');

    setMessages((previous) => [
      ...previous,
      {
        role: 'user',
        content: query
      },
      {
        role: 'assistant',
        loading: true
      }
    ]);

    setLoading(true);

    try {

      const data = await apiFetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          session_id: session,
          message: query
        })
      });

      setMessages((previous) => [
        ...previous.slice(0, -1),
        {
          role: 'assistant',
          content: data.answer,
          plan: data.plan,
          result: data.result
        }
      ]);

    } catch (error) {

      setMessages((previous) => [
        ...previous.slice(0, -1),
        {
          role: 'assistant',
          error: true,
          content: error.message
        }
      ]);

    } finally {

      setLoading(false);

    }
  };


  // ==========================================================
  // RENDER
  // ==========================================================

  return (
    <div className="app">

      <div className="orb orb1" />
      <div className="orb orb2" />
      <div className="orb orb3" />


      {/* =====================================================
          HEADER
      ===================================================== */}

      <header className="topbar">

        <div className="brand">

          <div className="brandMark">
            <Sparkles size={20} />
          </div>

          <div>

            <div className="brandName">
              Universal Dataset Intelligence
            </div>

            <div className="brandSub">
              Conversational Adaptive Analytics & Machine Learning
            </div>

          </div>

        </div>


        <div className="status">

          <span
            className={
              health?.gemini_enabled
                ? 'dot live'
                : 'dot'
            }
          />

          {
            health?.gemini_enabled
              ? 'Gemini Connected'
              : health?.backend_offline
                ? 'Backend Offline'
                : 'Gemini Not Connected'
          }

        </div>

      </header>


      {/* =====================================================
          MAIN
      ===================================================== */}

      <main className="shell">


        {/* ===================================================
            HERO
        =================================================== */}

        <section className="hero">

          <div className="heroTag">
            <span>●</span>
            AI-DRIVEN · VERIFIED ANALYTICS
          </div>

          <h1>
            Talk to your <em>data.</em>
            <br />
            Make better decisions.
          </h1>

          <p>
            Upload a dataset, ask questions naturally,
            and let UDI select the right analytical
            operation — from descriptive analytics to
            machine learning, forecasting and anomaly
            detection.
          </p>

          <div className="heroPills">

            <span>
              <ShieldCheck size={15} />
              Python is the source of truth
            </span>

            <span>
              <Brain size={15} />
              Gemini orchestrates
            </span>

            <span>
              <Activity size={15} />
              7 adaptive domains
            </span>

          </div>

        </section>


        {/* ===================================================
            MAIN LAYOUT
        =================================================== */}

        <div className="layout">


          {/* =================================================
              SIDEBAR
          ================================================= */}

          <aside className="sidebar">


            {/* DOMAIN PANEL */}

            <div className="panel">

              <div className="panelTitle">
                Domain
              </div>

              <div className="domainList">

                {domains.map((item) => (

                  <button
                    className={
                      'domain ' +
                      (domain === item.id
                        ? 'active'
                        : '')
                    }
                    key={item.id}
                    onClick={() =>
                      selectDomain(item.id)
                    }
                  >

                    <span className="domainIcon">
                      {item.icon}
                    </span>

                    <span>

                      <b>
                        {item.label}
                      </b>

                      <small>
                        {item.description}
                      </small>

                    </span>

                  </button>

                ))}

              </div>

            </div>


            {/* DATASET PANEL */}

            <div className="panel uploadPanel">

              <div className="panelTitle">
                Dataset
              </div>

              <button
                className="upload"
                onClick={() =>
                  fileRef.current?.click()
                }
                disabled={uploading}
              >

                <Upload size={18} />

                <span>
                  {
                    uploading
                      ? 'Loading…'
                      : 'Upload CSV / Excel'
                  }
                </span>

              </button>


              <input
                ref={fileRef}
                type="file"
                hidden
                accept=".csv,.xlsx,.xls"
                onChange={(event) =>
                  upload(
                    event.target.files?.[0]
                  )
                }
              />


              {profile && (

                <div className="profile">

                  <div className="profileHead">

                    <FileSpreadsheet size={15} />

                    <span>
                      {profile.dataset_name}
                    </span>

                  </div>


                  <div className="statGrid">

                    <div>
                      <b>
                        {profile.rows.toLocaleString()}
                      </b>
                      <small>
                        Rows
                      </small>
                    </div>

                    <div>
                      <b>
                        {profile.columns.length}
                      </b>
                      <small>
                        Columns
                      </small>
                    </div>

                    <div>
                      <b>
                        {profile.missing_total}
                      </b>
                      <small>
                        Missing
                      </small>
                    </div>

                    <div>
                      <b>
                        {profile.duplicate_rows}
                      </b>
                      <small>
                        Duplicates
                      </small>
                    </div>

                  </div>

                </div>

              )}

            </div>


            {/* NEW CHAT */}

            <button
              className="newChat"
              onClick={newChat}
            >

              <Plus size={17} />

              New Chat

            </button>

          </aside>


          {/* =================================================
              CHAT AREA
          ================================================= */}

          <section className="chatArea">


            {/* CHAT HEADER */}

            <div className="chatHeader">

              <div>

                <span className="eyebrow">
                  ACTIVE WORKSPACE
                </span>

                <h2>
                  {current.label} Intelligence
                </h2>

              </div>


              <div className="engine">

                <span className="pulse" />

                Adaptive engine live

              </div>

            </div>


            {/* EMPTY STATE */}

            {!profile &&
              messages.length === 0 && (

                <div className="empty">

                  <div className="emptyIcon">
                    <Sparkles size={28} />
                  </div>

                  <h3>
                    Start with your dataset
                  </h3>

                  <p>
                    Upload a CSV/XLSX file for
                    the selected domain. No fixed
                    target selection is required.
                  </p>


                  <div className="cards">

                    <div className="feature">

                      <Brain />

                      <b>
                        Ask naturally
                      </b>

                      <span>
                        “What is increasing?”
                        “Compare categories.”
                      </span>

                    </div>


                    <div className="feature">

                      <BarChart3 />

                      <b>
                        Verified analysis
                      </b>

                      <span>
                        Python computes every
                        number and chart.
                      </span>

                    </div>


                    <div className="feature">

                      <ShieldCheck />

                      <b>
                        Data sufficiency
                      </b>

                      <span>
                        Unsupported questions
                        are rejected safely.
                      </span>

                    </div>

                  </div>

                </div>

              )}


            {/* MESSAGES */}

            {messages.map((message, index) => (

              <Message
                key={index}
                m={message}
              />

            ))}


            {/* COMPOSER */}

            <div className="composer">

              <textarea
                value={input}
                onChange={(event) =>
                  setInput(event.target.value)
                }
                onKeyDown={(event) => {

                  if (
                    event.key === 'Enter' &&
                    !event.shiftKey
                  ) {

                    event.preventDefault();

                    ask();

                  }

                }}
                placeholder={
                  profile
                    ? 'Ask a question about your dataset…'
                    : 'Upload a dataset to begin…'
                }
                disabled={
                  !profile ||
                  loading
                }
              />


              <button
                onClick={() => ask()}
                disabled={
                  !profile ||
                  !input.trim() ||
                  loading
                }
              >

                <Send size={19} />

              </button>


              <div className="composerHint">

                Enter to send · Shift + Enter
                for a new line

              </div>

            </div>

          </section>

        </div>

      </main>

    </div>
  );
}


// ============================================================
// MESSAGE COMPONENT
// ============================================================

function Message({ m }) {

  if (m.loading) {

    return (

      <div className="msg assistant">

        <div className="avatar ai">
          <Bot size={16} />
        </div>

        <div className="bubble typing">

          <span />
          <span />
          <span />

        </div>

      </div>

    );
  }


  return (

    <div
      className={
        'msg ' +
        (m.role === 'user'
          ? 'user'
          : 'assistant')
      }
    >

      <div
        className={
          'avatar ' +
          (m.role === 'user'
            ? 'usr'
            : 'ai')
        }
      >

        {
          m.role === 'user'
            ? <User size={16} />
            : <Bot size={16} />
        }

      </div>


      <div className="messageBody">

        <div
          className={
            'bubble ' +
            (m.error ? 'error' : '')
          }
        >
          {m.content}
        </div>


        {m.result && (

          <ResultView
            result={m.result}
            plan={m.plan}
          />

        )}

      </div>

    </div>

  );
}


// ============================================================
// RESULT VIEW
// ============================================================

function ResultView({ result, plan }) {

  const [open, setOpen] =
    useState(false);


  const metrics =
    Object.entries(
      result?.metrics || {}
    ).slice(0, 4);


  return (

    <div className="result">


      {/* METRICS */}

      <div className="metrics">

        {metrics.map(([key, value]) => (

          <div
            className="metric"
            key={key}
          >

            <span>
              {key.replaceAll('_', ' ')}
            </span>

            <b>

              {
                typeof value === 'number'
                  ? (
                      Math.abs(value) >= 1000
                        ? value.toLocaleString(
                            undefined,
                            {
                              maximumFractionDigits: 2
                            }
                          )
                        : value.toFixed(2)
                    )
                  : String(value)
              }

            </b>

          </div>

        ))}

      </div>


      {/* CHART */}

      {result?.chart && (

        <Chart
          chart={result.chart}
        />

      )}


      {/* TABLE */}

      {result?.table?.length > 0 && (

        <div className="tableWrap">

          <table>

            <thead>

              <tr>

                {Object.keys(
                  result.table[0]
                )
                  .slice(0, 7)
                  .map((key) => (

                    <th key={key}>
                      {key}
                    </th>

                  ))}

              </tr>

            </thead>


            <tbody>

              {result.table
                .slice(0, 8)
                .map((row, index) => (

                  <tr key={index}>

                    {Object.keys(
                      result.table[0]
                    )
                      .slice(0, 7)
                      .map((key) => (

                        <td key={key}>

                          {
                            typeof row[key] ===
                            'number'
                              ? row[key].toLocaleString(
                                  undefined,
                                  {
                                    maximumFractionDigits: 2
                                  }
                                )
                              : String(
                                  row[key] ?? ''
                                )
                          }

                        </td>

                      ))}

                  </tr>

                ))}

            </tbody>

          </table>

        </div>

      )}


      {/* DETAILS */}

      <button
        className="details"
        onClick={() =>
          setOpen(!open)
        }
      >

        Analysis Details

        <ChevronDown
          size={15}
          className={
            open ? 'rot' : ''
          }
        />

      </button>


      {open && (

        <div className="detailsBox">

          <div>
            <b>
              Operation
            </b>

            <span>
              {plan?.operation || '—'}
            </span>
          </div>


          <div>
            <b>
              Target
            </b>

            <span>
              {plan?.target_column || '—'}
            </span>
          </div>


          <div>
            <b>
              Model
            </b>

            <span>
              {
                plan?.model_type ||
                result?.evaluation?.model ||
                'Deterministic analytics'
              }
            </span>
          </div>


          {result?.evaluation?.selected_model && (
            <div>
              <b>Selected Model</b>
              <span>{result.evaluation.selected_model}</span>
            </div>
          )}

          {result?.evaluation?.candidate_models?.length > 0 && (
            <div>
              <b>Models Evaluated</b>
              <span>{result.evaluation.candidate_models.length}</span>
            </div>
          )}

          {plan?.aggregation && (
            <div>
              <b>Aggregation</b>
              <span>{plan.aggregation}</span>
            </div>
          )}

          {plan?.time_column && (
            <div>
              <b>Time Column</b>
              <span>{plan.time_column}</span>
            </div>
          )}

          {plan?.reasoning && (
            <div>
              <b>Plan Reasoning</b>
              <span>{plan.reasoning}</span>
            </div>
          )}

          {result?.limitations?.map(
            (item, index) => (
              <p key={index}>
                • {item}
              </p>
            )
          )}

        </div>

      )}

    </div>

  );
}


// ============================================================
// CHART COMPONENT
// ============================================================

function Chart({ chart }) {

  const data =
    chart?.data || [];


  // ==========================================================
  // LINE CHART
  // ==========================================================

  if (chart.type === 'line') {

    return (

      <div className="chart">

        <ResponsiveContainer
          width="100%"
          height={260}
        >

          <LineChart data={data}>

            <CartesianGrid
              strokeDasharray="3 3"
              opacity={0.12}
            />

            <XAxis
              dataKey={chart.x}
            />

            <YAxis />

            <Tooltip />

            <Line
              type="monotone"
              dataKey={chart.y}
              stroke="#fbbf24"
              strokeWidth={3}
              dot={false}
            />

          </LineChart>

        </ResponsiveContainer>

      </div>

    );
  }


  // ==========================================================
  // BAR CHART
  // ==========================================================

  if (chart.type === 'bar') {

    return (

      <div className="chart">

        <ResponsiveContainer
          width="100%"
          height={260}
        >

          <BarChart data={data}>

            <CartesianGrid
              strokeDasharray="3 3"
              opacity={0.12}
            />

            <XAxis
              dataKey={chart.x}
            />

            <YAxis />

            <Tooltip />

            <Bar
              dataKey={chart.y}
              fill="#fb923c"
              radius={[
                6,
                6,
                0,
                0
              ]}
            />

          </BarChart>

        </ResponsiveContainer>

      </div>

    );
  }


  // ==========================================================
  // SCATTER CHART
  // ==========================================================

  if (chart.type === 'scatter') {

    /*
     * UDI supports dynamic scatter chart
     * column names.
     *
     * If x/y are supplied by backend,
     * use those. Otherwise preserve the
     * legacy row_index/value behavior.
     */

    const xKey =
      chart.x ||
      'row_index';

    const yKey =
      chart.y ||
      'value';


    return (

      <div className="chart">

        <ResponsiveContainer
          width="100%"
          height={260}
        >

          <ScatterChart>

            <CartesianGrid
              opacity={0.12}
            />

            <XAxis
              dataKey={xKey}
              name={xKey}
            />

            <YAxis
              dataKey={yKey}
              name={yKey}
            />

            <Tooltip />

            <Scatter
              data={data}
              fill="#a78bfa"
            />

          </ScatterChart>

        </ResponsiveContainer>

      </div>

    );
  }


  // ==========================================================
  // FORECAST CHART
  // ==========================================================

  if (chart.type === 'forecast') {

    const history =
      (chart.history || []).map(
        (item) => ({
          ...item,
          series: 'history'
        })
      );

    const forecast =
      (chart.forecast || []).map(
        (item) => ({
          ...item,
          series: 'forecast'
        })
      );


    return (

      <div className="chart">

        <ResponsiveContainer
          width="100%"
          height={260}
        >

          <LineChart
            data={[
              ...history,
              ...forecast
            ]}
          >

            <CartesianGrid
              strokeDasharray="3 3"
              opacity={0.12}
            />

            <XAxis
              dataKey="date"
            />

            <YAxis />

            <Tooltip />

            <Line
              type="monotone"
              dataKey="actual"
              stroke="#a78bfa"
              strokeWidth={2.5}
              dot={false}
            />

            <Line
              type="monotone"
              dataKey="forecast"
              stroke="#fbbf24"
              strokeWidth={3}
              dot={false}
            />

          </LineChart>

        </ResponsiveContainer>

      </div>

    );
  }


  return null;
}


// ============================================================
// APPLICATION ENTRY POINT
// ============================================================

const rootElement =
  document.getElementById('root');

createRoot(rootElement).render(
  <App />
);