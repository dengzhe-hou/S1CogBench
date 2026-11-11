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
console.log('Community board rendering passed, with separate provenance and retained undefined/infinite values.');
