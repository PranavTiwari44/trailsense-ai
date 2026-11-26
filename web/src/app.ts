// The TrailSense web page: send the question to POST /explain and show the answer.
// Build with `npm run build` in the web folder; the browser loads the compiled dist/app.js.

// ---------- The answer's shape (matches ExplainResponse in app/models.py) ----------

type Level = "Easy" | "Moderate" | "Hard" | "Dangerous" | "Unknown";

interface ExplainResponse {
  trail_name: string | null;
  level: Level;
  score: number | null;
  summary: string;
  reasons: string[];
  warnings: string[];
  gear_tips: string[];
  data_time: string | null;
  note: string;
}

/** An error from our API, with the HTTP status so we can explain it in plain words. */
class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

// ---------- Talking to the API ----------

/** Send the question to POST /explain and return the answer. Throws ApiError if the API says no. */
async function askQuestion(question: string): Promise<ExplainResponse> {
  const response = await fetch("/explain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: question }),
  });
  if (!response.ok) {
    throw new ApiError(response.status, await response.text());
  }
  return (await response.json()) as ExplainResponse;
}

/** Turn any error into a short message a normal person understands. */
function friendlyMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 503) {
      return "The AI isn't running right now. If this is your computer, start Ollama and try again.";
    }
    if (error.status === 502) {
      return "The AI got confused and couldn't give a proper answer. Please try asking again.";
    }
    if (error.status === 422) {
      return "Please type a question between 3 and 500 characters.";
    }
    return `Something went wrong on the server (error ${error.status}). Please try again.`;
  }
  return "Couldn't reach TrailSense. Check that the server is running, then try again.";
}

// ---------- Finding things on the page ----------

/** Get an element by its id, or stop with a clear message if the HTML doesn't have it. */
function getElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (element === null) {
    throw new Error(`The page is missing the element #${id}`);
  }
  return element as T;
}

const form = getElement<HTMLFormElement>("ask-form");
const input = getElement<HTMLInputElement>("question");
const askButton = getElement<HTMLButtonElement>("ask-button");
const loading = getElement<HTMLDivElement>("loading");
const loadingText = getElement<HTMLSpanElement>("loading-text");
const errorBox = getElement<HTMLDivElement>("error");
const result = getElement<HTMLElement>("result");

// ---------- Showing things ----------

/** Fill a list with items, and hide its whole section if there are none. */
function fillList(listId: string, sectionId: string, items: string[]): void {
  const list = getElement<HTMLUListElement>(listId);
  list.replaceChildren(); // remove the previous answer's items
  for (const item of items) {
    const li = document.createElement("li");
    li.textContent = item; // textContent, never innerHTML: AI text must not become HTML
    list.appendChild(li);
  }
  getElement<HTMLElement>(sectionId).hidden = items.length === 0;
}

/** "2026-11-25T21:15:00-05:00" -> "Weather data from Nov 25, 9:15 PM" in the reader's own time zone. */
function formatDataTime(text: string | null): string {
  if (text === null) {
    return "No weather data was available.";
  }
  const time = new Date(text);
  const shown = time.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
  return `Weather data from ${shown}`;
}

/** Put the answer on the page. */
function showResult(answer: ExplainResponse): void {
  getElement("trail-name").textContent = answer.trail_name ?? "Trail not found";

  const badge = getElement("level");
  badge.textContent = answer.level;
  badge.className = `badge ${answer.level.toLowerCase()}`; // colour comes from the CSS class

  const score = getElement("score");
  if (answer.score === null) {
    score.textContent = "";
  } else {
    score.textContent = `Score ${answer.score}/100`;
  }

  getElement("summary").textContent = answer.summary;
  // No trail found: remind people they can name a city outside Ottawa–Gatineau.
  getElement("not-found-tip").hidden = answer.trail_name !== null;
  fillList("reasons", "reasons-box", answer.reasons);
  fillList("warnings", "warnings-box", answer.warnings);
  fillList("gear", "gear-box", answer.gear_tips);
  getElement("data-time").textContent = formatDataTime(answer.data_time);
  getElement("note").textContent = answer.note;

  result.hidden = false;
}

/** Show an error message instead of an answer. */
function showError(message: string): void {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

// ---------- Loading state, with a seconds counter (the AI takes 5 to 60 s) ----------

let timerId: number | undefined;

/** Turn the loading state on or off: spinner, counter, and a disabled button. */
function setLoading(on: boolean): void {
  askButton.disabled = on;
  loading.hidden = !on;
  if (on) {
    const started = Date.now();
    loadingText.textContent = "Checking the trail, weather and elevation…";
    timerId = window.setInterval(() => {
      const seconds = Math.round((Date.now() - started) / 1000);
      loadingText.textContent = `Checking the trail, weather and elevation… ${seconds} s`;
    }, 1000);
  } else {
    window.clearInterval(timerId);
  }
}

// ---------- What happens when someone asks ----------

/** Ask the question in the box and show the answer or a friendly error. */
async function handleAsk(): Promise<void> {
  const question = input.value.trim();
  if (question.length < 3) {
    showError("Please type a question about a trail, e.g. “How hard is the Wolf Trail right now?”");
    return;
  }

  errorBox.hidden = true;
  result.hidden = true;
  setLoading(true);
  try {
    showResult(await askQuestion(question));
  } catch (error) {
    showError(friendlyMessage(error));
  } finally {
    setLoading(false);
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault(); // stop the browser from reloading the page
  handleAsk();
});

// Clicking an example fills the box and asks straight away.
for (const button of document.querySelectorAll<HTMLButtonElement>(".example")) {
  button.addEventListener("click", () => {
    input.value = button.textContent ?? "";
    handleAsk();
  });
}
