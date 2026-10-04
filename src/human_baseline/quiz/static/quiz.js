// ASL Sign Quiz — all the page logic lives here.
//
// A "question" is one entry from practice.json or /test.json:
//   { "text": "no", "label": 4, "clip_path": "msasl/no/xyz.mp4", ... }
// "text" is the correct answer; "clip_path" is the video to play.
// Test questions also have "video_id" (a001...), "box" (signer area to crop to) and
// "covers" (rectangles hiding on-screen text) — see test_list() in app.py.

// ---------- State ----------
let mode = "practice"; // "practice" or "test"
let questions = [];    // shuffled list of entries for the current run
let current = 0;       // index of the question being shown
let score = 0;         // number of correct answers so far

// Test mode only
let playerName = "";
let results = [];          // one object per answered question (becomes one CSV row)
let testStartTime = 0;     // performance.now() when the test started
let questionStartTime = 0; // performance.now() when the current video appeared
let timerInterval = null;  // updates the on-screen clock

// ---------- Page elements ----------
const $ = (id) => document.getElementById(id);
const video = $("video");
const answerInput = $("answer-input");

// ---------- Helpers ----------

// Show one screen (<section class="screen">) and hide the others.
function showScreen(id) {
  document.querySelectorAll(".screen").forEach((s) => (s.hidden = s.id !== id));
}

// Return a shuffled copy of an array (Fisher–Yates shuffle).
function shuffle(array) {
  const a = array.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// Answers are compared ignoring case and surrounding spaces: " Hello " == "hello".
function normalize(text) {
  return text.trim().toLowerCase();
}

// 83.4 seconds -> "1:23"
function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

// Local date/time as "2026-09-30 14:05:09" (for the CSV).
function localTimestamp() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ` +
         `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

// ---------- Video display ----------

// Crop the video to the question's "box" (the signer) and draw its caption covers.
// Practice questions have no box, so the whole frame is shown.
function layoutVideo() {
  const q = questions[current];
  const [top, left, bottom, right] = q.box || [0, 0, 1, 1];
  const cropW = right - left;
  const cropH = bottom - top;

  // The visible window gets the crop's shape (in pixels of the real video).
  const ratio = (cropW * video.videoWidth) / (cropH * video.videoHeight);
  const box = $("video-box");
  box.style.aspectRatio = ratio;
  box.style.width = `min(100%, calc(60vh * ${ratio}))`;

  // Scale and shift the full frame so the crop area exactly fills the window.
  const frame = $("video-frame");
  frame.style.width = `${100 / cropW}%`;
  frame.style.height = `${100 / cropH}%`;
  frame.style.left = `${(-left / cropW) * 100}%`;
  frame.style.top = `${(-top / cropH) * 100}%`;
  frame.style.right = frame.style.bottom = "auto";

  // Caption covers are in full-frame fractions, so they go inside the frame.
  frame.querySelectorAll(".caption-cover").forEach((c) => c.remove());
  for (const [l, t, r, b] of q.covers || []) {
    const cover = document.createElement("div");
    cover.className = "caption-cover";
    cover.style.left = `${l * 100}%`;
    cover.style.top = `${t * 100}%`;
    cover.style.width = `${(r - l) * 100}%`;
    cover.style.height = `${(b - t) * 100}%`;
    frame.append(cover);
  }
}

function replayVideo() {
  video.currentTime = 0;
  video.play().catch(() => {});
}

// ---------- Starting a run ----------

async function startPractice() {
  mode = "practice";
  const response = await fetch("/practice.json");
  startRun(await response.json());
}

async function startTest(event) {
  event.preventDefault(); // stop the form from reloading the page
  playerName = $("name-input").value.trim();
  if (playerName === "") return;

  mode = "test";
  const response = await fetch("/test.json");
  results = [];
  startRun(await response.json());

  // Start the visible clock: time on the current video, plus the total so far.
  testStartTime = performance.now();
  $("timer").hidden = false;
  updateTimer();
  timerInterval = setInterval(updateTimer, 100);
}

function updateTimer() {
  const now = performance.now();
  const videoSeconds = (now - questionStartTime) / 1000;
  const totalSeconds = (now - testStartTime) / 1000;
  $("timer").textContent =
    `This video: ${videoSeconds.toFixed(1)} s  ·  Total: ${formatTime(totalSeconds)}`;
}

function startRun(entries) {
  questions = shuffle(entries);
  current = 0;
  score = 0;
  $("timer").hidden = mode !== "test";
  $("skip-btn").hidden = mode !== "test";
  showScreen("quiz-screen");
  showQuestion();
}

// ---------- One question ----------

function showQuestion() {
  const q = questions[current];
  $("progress").textContent = `Question ${current + 1} of ${questions.length}`;

  // Load and autoplay the video (layoutVideo runs once its size is known).
  $("video-missing").hidden = true;
  video.src = "/" + q.clip_path;
  video.play().catch(() => {}); // autoplay can be blocked; the user can click Replay

  // Reset the answer area.
  answerInput.value = "";
  answerInput.disabled = false;
  $("submit-btn").hidden = false;
  $("next-btn").hidden = true;
  $("feedback").textContent = "";
  $("feedback").className = "";
  answerInput.focus();

  questionStartTime = performance.now(); // time on this question starts now
}

function submitAnswer(event) {
  event.preventDefault(); // stop the form from reloading the page
  if (normalize(answerInput.value) === "") return; // ignore empty submissions
  handleAnswer(answerInput.value);
}

// Called with the typed answer, or "" when the user clicks "Don't know".
function handleAnswer(rawAnswer) {
  const q = questions[current];
  const isCorrect = normalize(rawAnswer) === normalize(q.text);
  if (isCorrect) score++;

  if (mode === "test") {
    // Test mode: record the answer silently and go straight to the next video.
    results.push({
      participant: playerName,
      timestamp: localTimestamp(),
      question_number: current + 1,
      video_id: q.video_id,
      video_file: q.clip_path,
      correct_answer: q.text,
      user_answer: rawAnswer.trim(),
      is_correct: isCorrect ? 1 : 0,
      time_spent_seconds: ((performance.now() - questionStartTime) / 1000).toFixed(3),
    });
    nextQuestion();
    return;
  }

  // Practice mode: tell the user right away.
  const feedback = $("feedback");
  if (isCorrect) {
    feedback.textContent = "Correct!";
    feedback.className = "correct";
  } else {
    feedback.textContent = `Wrong — the answer is "${q.text}"`;
    feedback.className = "wrong";
  }

  // Lock the answer and show the Next button (pressing Enter now clicks Next).
  answerInput.disabled = true;
  $("submit-btn").hidden = true;
  $("next-btn").hidden = false;
  $("next-btn").focus();
}

function nextQuestion() {
  current++;
  if (current < questions.length) {
    showQuestion();
  } else {
    finishRun();
  }
}

// ---------- End of a run ----------

function finishRun() {
  video.pause();
  $("final-score").textContent = `${score} / ${questions.length}`;
  $("restart-btn").hidden = mode !== "practice";
  $("final-time").hidden = mode !== "test";
  $("save-status").textContent = "";
  $("retry-save-btn").hidden = true;
  showScreen("end-screen");

  if (mode === "test") {
    clearInterval(timerInterval);
    const totalSeconds = (performance.now() - testStartTime) / 1000;
    $("final-time").textContent =
      `Total time: ${formatTime(totalSeconds)} ` +
      `(average ${(totalSeconds / questions.length).toFixed(1)} s per video)`;
    saveResults();
  }
}

// Send the test results to the server, which appends them to the CSV file.
async function saveResults() {
  const status = $("save-status");
  status.className = "";
  status.textContent = "Saving results…";
  $("retry-save-btn").hidden = true;
  try {
    const response = await fetch("/api/results", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rows: results }),
    });
    if (!response.ok) throw new Error(`server said ${response.status}`);
    const info = await response.json();
    status.textContent = `Saved ${info.saved} answers to ${info.file}`;
    status.className = "correct";
    results = []; // saved — nothing left to lose
  } catch (err) {
    status.textContent = `Could not save results (${err.message}). Keep this page open and click Retry.`;
    status.className = "wrong";
    $("retry-save-btn").hidden = false;
  }
}

// ---------- Wire up buttons ----------

$("practice-btn").addEventListener("click", startPractice);
$("restart-btn").addEventListener("click", startPractice);
$("test-form").addEventListener("submit", startTest);
$("home-btn").addEventListener("click", () => showScreen("start-screen"));
$("answer-form").addEventListener("submit", submitAnswer);
$("skip-btn").addEventListener("click", () => handleAnswer(""));
$("next-btn").addEventListener("click", nextQuestion);
$("retry-save-btn").addEventListener("click", saveResults);
$("replay-btn").addEventListener("click", replayVideo);
$("video-box").addEventListener("click", replayVideo);

// Once a video's size is known, crop it to the right shape.
video.addEventListener("loadedmetadata", layoutVideo);

// If a video file doesn't exist, say so instead of showing a blank player.
// The question can still be answered.
video.addEventListener("error", () => {
  const missing = $("video-missing");
  missing.textContent = `Video not found: ${questions[current].clip_path}`;
  missing.hidden = false;
});

// Warn before closing/reloading the tab while a test is running or unsaved.
window.addEventListener("beforeunload", (event) => {
  const testRunning = mode === "test" && !$("quiz-screen").hidden;
  if (testRunning || results.length > 0) event.preventDefault();
});
