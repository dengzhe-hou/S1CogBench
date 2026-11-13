// UI rendering fixtures, not model results. Run with node tests/test_community_web.cjs.
const assert = require('node:assert/strict');
const {communityHTML, communityValue} = require('../homepage/community.js');
const board = {
  context: {metric: 'human_nll', unit: 'nats_per_response', aggregation: 'participant_mean', target: 'human_choice'},
  direction: 'lower',
  entries: [
    {model_label: 'Paper model', value: .5, value_raw: '.5', value_status: 'finite', n: 75, rank: 1,
      origin: 'paper', evidence_url: 'https://arxiv.org/abs/2611.00911'},
    {model_label: 'Fixture <model>', value: 'Infinity', value_raw: 'Infinity', value_status: 'defined_positive_infinity',
      n: 75, rank: 2, origin: 'community', evidence_url: 'https://example.org/fixture',
      metadata: {prompt: 'actual prompt fixture', readout: 'fixture readout'}}
  ]
};
const html = communityHTML(board);
assert.match(html, /Paper snapshot/);
assert.match(html, /Reviewed community report/);
assert.match(html, /Fixture &lt;model&gt;/);
assert.match(html, /actual prompt fixture/);
assert.match(html, /human_nll ↓/);
assert.match(html, /\+∞/);
assert.equal(communityValue({value: null, value_status: 'undefined'}), 'Undefined');
board.entries[1].rank = null;
assert.match(communityHTML(board), /Unranked/);
assert.doesNotMatch(communityHTML(board), /Extension task/);
assert.match(communityHTML({...board, track: 'extension'}), /Extension task, contributed outside the 24-task paper suite/);
console.log('Community board rendering passed, with separate provenance and retained undefined/infinite values.');

const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '..');
const mainSource = fs.readFileSync(path.join(root, 'homepage/app.js'), 'utf8');
const mainTable = JSON.parse(fs.readFileSync(path.join(root, 'results/latest/table2.json'), 'utf8'));
const elements = new Map();
const element = selector => {
  if (!elements.has(selector)) elements.set(selector, {
    innerHTML: '', textContent: '', value: '', checked: false,
    dataset: {}, options: [{}, {}], querySelectorAll: () => [], addEventListener: () => {}
  });
  return elements.get(selector);
};
element('#leaderboard-model').value = 'all';
const context = {document: {querySelector: element}, mainTable, Intl};
vm.createContext(context);
vm.runInContext(mainSource.slice(0, mainSource.indexOf('async function initialize()')), context);
const render = () => {
  vm.runInContext('renderLeaderboard(mainTable);', context);
  return element('#leaderboard-body').innerHTML.match(/<tr>[\s\S]*?<\/tr>/g);
};
let renderedRows = render();
assert.equal(renderedRows.length, 27);
for (const row of renderedRows) assert.equal(row.match(/<td(?:\s|>)/g).length, 12);
for (const model of ['Laya', 'Intern', 'Llama', 'Minitaur', 'Qwen', 'Nox', 'CLM']) {
  assert.match(element('#results thead').innerHTML, new RegExp('>' + model + '<'));
}
assert.doesNotMatch(element('#results thead').innerHTML, />Haiku</);
element('#leaderboard-api').checked = true;
renderedRows = render();
for (const row of renderedRows) assert.equal(row.match(/<td(?:\s|>)/g).length, 16);
assert.match(element('#results thead').innerHTML, />Haiku</);
element('#leaderboard-api').checked = false;
element('#leaderboard-model').value = 'Nox';
renderedRows = render();
for (const row of renderedRows) assert.equal(row.match(/<td(?:\s|>)/g).length, 6);
element('#leaderboard-model').value = 'all';
element('#leaderboard-search').value = 'mouselab';
assert.equal(render().length, 2);
element('#leaderboard-search').value = '';
element('#leaderboard-domain').value = 'Evidence';
assert.equal(render().length, 4);
const page = fs.readFileSync(path.join(root, 'homepage/index.html'), 'utf8');
assert.match(page, /colspan="7" scope="colgroup">Full-suite open models/);
assert.match(page, /<th scope="col">Nox<\/th><th scope="col">CLM<\/th>/);
console.log('Main-table rendering passed for all 27 modes, seven models, API columns, model selection, domain and task search.');
