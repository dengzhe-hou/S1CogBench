/* The merged community registry is a static build output. No browser upload occurs. */
const escapeCommunity = value => String(value ?? "").replace(/[&<>"']/g, c => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
}[c]));

function communityValue(row) {
  if (row.value_status === "undefined") return "Undefined";
  if (row.value_status === "defined_positive_infinity") return "+∞";
  return new Intl.NumberFormat("en", {maximumFractionDigits: 6}).format(row.value);
}

function communityHTML(board) {
  const esc = escapeCommunity;
  const context = board.context;
  const arrow = board.direction === "lower" ? " ↓" : board.direction === "higher" ? " ↑" : "";
  return '<p class="smallprint">' + esc(context.metric + arrow) + ' · ' + esc(context.unit) + ' · ' +
    esc(context.aggregation) + '</p><p class="smallprint">' + esc(context.target) + '</p>' +
    '<div class="table-scroll"><table class="result-table"><thead><tr><th>Rank</th><th>Model and readout</th>' +
    '<th>Value' + arrow + '</th><th>n</th><th>Evidence</th></tr></thead><tbody>' +
    board.entries.map(row => {
      const metadata = row.metadata;
      const details = metadata ? '<details><summary>Run settings</summary>' +
        Object.entries(metadata).filter(([key]) => key !== "evidence_url").map(([key, value]) =>
          '<p><strong>' + esc(key.replaceAll("_", " ")) + '</strong> ' + esc(value) + '</p>').join("") +
        '</details>' : "";
      return '<tr><td>' + (row.rank ?? "Unranked") + '</td><td>' + esc(row.model_label) +
        '<small class="result-meta">' + (row.origin === "community" ? "Reviewed community report" : row.origin === "maintainer" ? "Maintainer results" : "Paper snapshot") +
        '</small>' + details + '</td><td title="' + esc(row.value_raw) + '">' + esc(communityValue(row)) +
        '</td><td>' + esc(row.n ?? row.n_raw) + '</td><td><a href="' + esc(row.evidence_url) +
        '" target="_blank" rel="noopener">' + (metadata ? "Run evidence" : "Paper") + '</a></td></tr>';
    }).join("") + '</tbody></table></div>';
}

async function initializeCommunity() {
  const status = document.querySelector("#community-status");
  try {
    const response = await fetch("data/community.json");
    if (!response.ok) throw Error("Community results could not be loaded.");
    const data = await response.json();
    const select = document.querySelector("#community-context");
    select.innerHTML = data.boards.map((board, i) => '<option value="' + i + '">' +
      escapeCommunity(board.label + " · " + board.context.protocol) + '</option>').join("");
    const render = () => {
      const board = data.boards[Number(select.value)];
      document.querySelector("#community-results").innerHTML = board ? communityHTML(board) :
        '<p>No result rows in this protocol.</p>';
    };
    select.addEventListener("change", render);
    status.textContent = data.submission_count ? data.submission_count + " merged community submissions. " + data.review_policy :
      "Open for submissions. No external contribution has been merged yet. Seven-model maintainer results are shown below.";
    render();
  } catch (error) {
    status.textContent = error.message + " You can still read the paper results and submit through GitHub.";
  }
}

if (typeof module !== "undefined") module.exports = {communityHTML, communityValue};
if (typeof document !== "undefined") initializeCommunity();
