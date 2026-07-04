// ---- CONFIGURE THIS: your live Cloud Run backend URL ----
const API_URL = "https://gpt-chess-backend-513208374732.us-central1.run.app/generate";

const board = Chessboard('board', {
  position: 'start',
  pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png'
});

console.log('Chess loaded?', typeof Chess);

const strategyEl = document.getElementById('strategy');
const tempField = document.getElementById('tempField');
const kField = document.getElementById('kField');
const pField = document.getElementById('pField');

function updateVisibleFields(){
  const s = strategyEl.value;
  tempField.style.display = (s === 'temperature' || s === 'top_k' || s === 'top_p') ? 'block' : 'none';
  kField.style.display = (s === 'top_k') ? 'block' : 'none';
  pField.style.display = (s === 'top_p') ? 'block' : 'none';
}
strategyEl.addEventListener('change', updateVisibleFields);
updateVisibleFields();

const sliderLabels = { temperature: 'tempVal', k: 'kVal', p: 'pVal' };
Object.entries(sliderLabels).forEach(([id, valId]) => {
  const el = document.getElementById(id);
  const val = document.getElementById(valId);
  el.addEventListener('input', () => val.textContent = el.value);
});

let legalMoves = [];
let firstIllegal = null;
let playIndex = 0;
let playTimer = null;

const btnGenerate = document.getElementById('btnGenerate');
const btnPlay = document.getElementById('btnPlay');
const btnBack = document.getElementById('btnBack');
const btnFwd = document.getElementById('btnFwd');
const speedEl = document.getElementById('speed');
const scoresheetBody = document.getElementById('scoresheetBody');
const illegalFlag = document.getElementById('illegalFlag');

function extractMoves(text){
  return text.split(/\s+/).filter(tok => tok.length && !tok.endsWith('.'));
}

function renderScoresheet(allMoves, illegalIndex){
  let html = '';
  for (let i = 0; i < allMoves.length; i += 2){
    const num = Math.floor(i / 2) + 1;
    const w = allMoves[i] || '';
    const b = allMoves[i + 1] || '';
    const wIllegal = illegalIndex === i;
    const bIllegal = illegalIndex === i + 1;
    html += `<div class="scoresheet-row" data-idx="${i}">
      <span class="num">${num}</span>
      <span class="mv ${wIllegal ? 'illegal-mv' : ''}">${w}</span>
      <span class="mv ${bIllegal ? 'illegal-mv' : ''}">${b}</span>
    </div>`;
  }
  scoresheetBody.innerHTML = html || '<div class="empty-note">No moves generated.</div>';
}

function highlightRow(moveIdx){
  document.querySelectorAll('.scoresheet-row').forEach(r => r.classList.remove('active'));
  const rowIdx = Math.floor(moveIdx / 2) * 2;
  const row = document.querySelector(`.scoresheet-row[data-idx="${rowIdx}"]`);
  if (row) { row.classList.add('active'); row.scrollIntoView({ block: 'nearest' }); }
}

function setPlaybackEnabled(on){
  btnPlay.disabled = !on; btnBack.disabled = !on; btnFwd.disabled = !on;
}

function goToIndex(idx){
  const game = new Chess();
  for (let i = 0; i <= idx; i++){ game.move(legalMoves[i]); }
  board.position(game.fen());
  highlightRow(idx);
  playIndex = idx;
}

btnFwd.addEventListener('click', () => {
  if (playIndex < legalMoves.length - 1) goToIndex(playIndex + 1);
});

btnBack.addEventListener('click', () => {
  if (playIndex > 0) goToIndex(playIndex - 1);
});

btnPlay.addEventListener('click', () => {
  if (playTimer){
    clearInterval(playTimer); playTimer = null; btnPlay.textContent = '▶ play';
    return;
  }
  if (playIndex >= legalMoves.length - 1){ goToIndex(0); }
  btnPlay.textContent = '❙❙ pause';
  playTimer = setInterval(() => {
    if (playIndex >= legalMoves.length - 1){
      clearInterval(playTimer); playTimer = null; btnPlay.textContent = '▶ play';
      return;
    }
    goToIndex(playIndex + 1);
  }, parseInt(speedEl.value));
});

btnGenerate.addEventListener('click', async () => {
  clearInterval(playTimer); playTimer = null;
  btnGenerate.disabled = true;
  btnGenerate.textContent = 'Generating…';
  illegalFlag.style.display = 'none';
  illegalFlag.classList.remove('flag-success');
  setPlaybackEnabled(false);
  board.position('start');

  const body = {
    prompt: document.getElementById('prompt').value || '1. e4',
    strategy: strategyEl.value,
    max_new_tokens: 70,
    temperature: parseFloat(document.getElementById('temperature').value),
    k: parseInt(document.getElementById('k').value),
    p: parseFloat(document.getElementById('p').value)
  };

  try {
    const res = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const data = await res.json();
    const allMoves = extractMoves(data.generated_text);

    const game = new Chess();
    legalMoves = [];
    firstIllegal = null;
    for (let i = 0; i < allMoves.length; i++){
      const result = game.move(allMoves[i], { sloppy: true });
      if (!result){ firstIllegal = i; break; }
      legalMoves.push(allMoves[i]);
    }

    renderScoresheet(allMoves, firstIllegal);

    if (firstIllegal !== null){
      illegalFlag.style.display = 'block';
      illegalFlag.textContent = `Model played an illegal move at move ${firstIllegal + 1} ("${allMoves[firstIllegal]}") — generation stopped there. This is expected: the model has no explicit board-state tracking.`;
    } else {
      illegalFlag.style.display = 'block';
      illegalFlag.classList.add('flag-success');
      illegalFlag.textContent = `All ${legalMoves.length} generated moves were legal — the requested move limit was reached with no errors.`;
    }

    if (legalMoves.length > 0){
        setPlaybackEnabled(true);
        goToIndex(0);
        btnPlay.click();
    }
  } catch (err) {
    console.error('Generation error:', err);
    illegalFlag.style.display = 'block';
    illegalFlag.textContent = 'Error: ' + err.message;
  }

  btnGenerate.disabled = false;
  btnGenerate.textContent = 'Generate game';
});