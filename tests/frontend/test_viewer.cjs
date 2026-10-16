'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const viewerSource = fs.readFileSync(
  path.join(__dirname, '../../src/visualmap/visualization/viewer.js'), 'utf8');

function element(tag = 'div') {
  const item = {
    tagName: tag,
    textContent: '',
    children: [],
    style: {},
    attributes: {},
    events: {},
    options: [],
    hidden: false,
    checked: true,
    disabled: false,
    width: 600,
    height: 400,
    clientWidth: 600,
    clientHeight: 400,
    value: '',
    currentTime: 0,
    duration: 4,
    paused: true,
    ended: false,
    videoWidth: 100,
    videoHeight: 80,
    classList: {toggle() {}},
    append(...nodes) { this.children.push(...nodes); },
    replaceChildren(...nodes) { this.children = nodes; this.textContent = ''; },
    add(option) { this.options.push(option); if (this.options.length === 1) this.value = option.value; },
    setAttribute(name, value) { this.attributes[name] = value; },
    addEventListener(name, callback) { this.events[name] = callback; },
    setPointerCapture() {},
    releasePointerCapture() {},
    getBoundingClientRect() { return {left: 0, top: 0}; },
    play() { this.paused = false; return Promise.resolve(); },
    pause() { this.paused = true; },
  };
  let selectedIndex = 0;
  Object.defineProperty(item, 'selectedIndex', {
    get: () => selectedIndex,
    set(value) { selectedIndex = value; item.value = item.options[value]?.value ?? ''; },
  });
  const drawing = {
    calls: [],
    measureText: text => ({width: String(text).length * 7}),
    createLinearGradient: () => ({addColorStop() {}}),
  };
  for (const method of ['setTransform', 'fillRect', 'clearRect', 'strokeRect', 'beginPath',
    'arc', 'stroke', 'fill', 'moveTo', 'lineTo', 'closePath', 'save', 'restore', 'setLineDash', 'fillText']) {
    drawing[method] = (...args) => drawing.calls.push({method, args});
  }
  item.getContext = () => drawing;
  return item;
}

function harness(overrides = {}, globals = {}) {
  const elements = new Map();
  const get = id => {
    if (!elements.has(id)) elements.set(id, element(id));
    return elements.get(id);
  };
  const identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const DATA = {
    trajectory: [[0, 0, 0], [0, 0, 1]],
    trajectory_timestamps: [0, 1],
    trajectory_frame_ids: [0, 30],
    trajectory_rotations: [identity, identity],
    map_points: [],
    objects: [],
    object_observations: [],
    summary: {
      detector: 'yolox_onnx',
      video: {fps: 30},
      intrinsics: {fx: 100, fy: 100, cx: 50, cy: 40, width: 100, height: 80},
    },
    ...overrides,
  };
  const context = vm.createContext({
    DATA,
    document: {
      getElementById: get,
      createElement: element,
      createTextNode: text => ({textContent: String(text), children: []}),
    },
    window: {devicePixelRatio: 1, addEventListener() {}},
    Option: function Option(text, value) { this.text = text; this.value = value; },
    requestAnimationFrame() {},
    console,
    ...globals,
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../../src/visualmap/visualization/viewer_occupancy.js'),'utf8'), context);
  vm.runInContext(viewerSource, context, {filename: 'viewer.js'});
  return {context, elements, get, run: source => vm.runInContext(source, context)};
}

function textOf(item) {
  return [item.textContent, ...(item.children || []).map(textOf)].filter(Boolean).join(' ');
}

function storedWindow(values = new Map()) {
  return {devicePixelRatio: 1, addEventListener() {}, localStorage: {
    getItem: key => values.get(key), setItem: (key, value) => values.set(key, value),
  }};
}

function selectReviewCrops(app, times = [0, 1, 2]) {
  app.get('llmStart').events.click();
  for (const time of times) {
    app.get('sourceVideo').currentTime = time;
    app.get('llmCapture').events.click();
    app.get('videoOverlay').events.pointerdown({button: 0, pointerId: 1,
      clientX: 100, clientY: 50, preventDefault() {}});
    app.get('videoOverlay').events.pointerup({pointerId: 1, clientX: 200, clientY: 150});
  }
}

function reviewApi(calls, reviewResponse, submission = {id: 'retry-review', status: 'processing'}) {
  return async (url, options = {}) => {
    calls.push({url, ...options});
    let payload;
    if (url.startsWith('/api/llm/context')) payload = {token: 'local-test-token', settings: {
      base_url: 'https://vision.example/v1', model: 'vision-model', has_api_key: true,
    }};
    else if (url.startsWith('/api/llm/labels?')) payload = {labels: []};
    else if (url === '/api/llm/reviews') payload = submission;
    else if (url === '/api/llm/reviews/retry-review') {
      if (reviewResponse instanceof Error) throw reviewResponse;
      payload = reviewResponse;
    } else throw new Error('Unexpected mocked request: ' + url);
    return {ok: true, json: async () => payload};
  };
}

const settleAsync = () => new Promise(resolve => setImmediate(resolve));
const filledCrops = app => app.get('llmEvidence').children.filter(node => node.className?.includes('filled'));
const optionText = (app, select, track) => app.get(select).options.find(option => option.value === String(track))?.text;

const plain = value => JSON.parse(JSON.stringify(value));
const close = (actual, expected, tolerance = 1e-9) =>
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} differs from ${expected}`);

test('front projection keeps OpenCV right/right and down/down, rejects points behind observer', () => {
  const app = harness();
  const projected = app.run(`
    mode = 'perspective';
    const testBasis = lookAtBasis([0,0,-10], [0,0,0]);
    [project([0,0,0],testBasis,600,400),
     project([1,0,0],testBasis,600,400),
     project([0,1,0],testBasis,600,400),
     project([0,0,-11],testBasis,600,400)]
  `);
  assert.ok(projected[1].x > projected[0].x, 'world X right must stay on the right');
  assert.ok(projected[2].y > projected[0].y, 'world Y down must stay below');
  assert.equal(projected[3], null);
});

test('top and side presets retain the declared X/Z and Z/Y directions', () => {
  const app = harness();
  const top = app.run(`
    setView('top');
    const topBasis=lookAtBasis(orbitEye(),target);
    [project([0,0,0],topBasis,600,400),project([1,0,0],topBasis,600,400),
     project([0,0,1],topBasis,600,400)]
  `);
  assert.ok(top[1].x > top[0].x);
  assert.ok(top[2].y < top[0].y, 'top preset world Z must point up, like the plan');
  const side = app.run(`
    setView('side');
    const sideBasis=lookAtBasis(orbitEye(),target);
    [project([0,0,0],sideBasis,600,400),project([0,0,1],sideBasis,600,400),
     project([0,1,0],sideBasis,600,400)]
  `);
  assert.ok(side[1].x > side[0].x, 'side preset Z must point right');
  assert.ok(side[2].y > side[0].y, 'side preset Y must point down');
});

test('frustum follows a rotated camera optical axis, with translation applied once', () => {
  const app = harness();
  const segments = app.run(`
    const testSegments=[];
    line=(a,b)=>testSegments.push([a,b]);
    drawFrustum({p:[2,3,4],R:[[0,0,1],[0,1,0],[-1,0,0]]},
      lookAtBasis([2,3,-10],[2,3,4]),600,400);
    testSegments
  `);
  assert.equal(segments.length, 9);
  const [start, end] = segments.at(-1);
  assert.deepEqual(plain(start), [2, 3, 4]);
  assert.ok(end[0] > start[0], 'camera local +Z rotates to world +X');
  close(end[1], start[1]);
  close(end[2], start[2]);
  const ray = segments[0][1].map((value, i) => value - start[i]);
  assert.ok(ray[0] > 0 && ray[1] < 0 && ray[2] > 0,
    'top-left pixel ray rotates with the camera rather than mirroring world X');
});

test('frustum corners use calibrated principal point rather than assuming image centre', () => {
  const app = harness({summary: {intrinsics: {
    fx: 100, fy: 100, cx: 30, cy: 20, width: 100, height: 80,
  }}});
  const ray = app.run(`
    const calibratedSegments=[];
    line=(a,b)=>calibratedSegments.push([a,b]);
    drawFrustum({p:[0,0,0],R:[[1,0,0],[0,1,0],[0,0,1]]},
      lookAtBasis([0,0,-10],[0,0,0]),600,400);
    calibratedSegments[0][1]
  `);
  close(ray[0] / ray[2], -0.3);
  close(ray[1] / ray[2], -0.2);
});

test('camera frustum pitch, roll and half-turn have the expected physical directions', () => {
  const cases = [
    {name: 'pitch', rotation: [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
      forward: [0, -1, 0], topLeftSigns: [-1, -1, -1]},
    {name: 'roll', rotation: [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
      forward: [0, 0, 1], topLeftSigns: [1, -1, 1]},
    {name: 'half-turn', rotation: [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
      forward: [0, 0, -1], topLeftSigns: [1, -1, -1]},
  ];
  for (const scenario of cases) {
    const app = harness();
    app.context.testRotation = scenario.rotation;
    const segments = app.run(`
      const directionSegments=[];
      line=(a,b)=>directionSegments.push([a,b]);
      drawFrustum({p:[0,0,0],R:testRotation},lookAtBasis([0,0,-10],[0,0,0]),600,400);
      directionSegments
    `);
    const direction = segments.at(-1)[1];
    const length = Math.hypot(...direction);
    for (let axis = 0; axis < 3; axis++) {
      close(direction[axis] / length, scenario.forward[axis]);
      assert.equal(Math.sign(segments[0][1][axis]), scenario.topLeftSigns[axis], scenario.name);
    }
  }
});

function temporalObjects() {
  return [
    {track_id: 1, class_id: 39, class_name: 'bottle', confidence: 0.9, observations: 4,
      frame_id: 30, xyz: [100, 0, 5], position_frame: 30, position_uncertainty: 10,
      position_status: 'estimated', position_source: 'sparse_observed',
      classification_consistency: 0.8, classification_observations: 3,
      classification_evidence: {latest_raw_class: 'cup', latest_raw_confidence: 0.8,
        alternatives: [{class_id: 41, class_name: 'cup', confidence: 0.6}]},
      position_history: [{frame_id: 0, timestamp: 0, xyz: [0, 0, 5], uncertainty: 0.1,
        source: 'sparse_observed', status: 'estimated'},
      {frame_id: 30, timestamp: 1, xyz: [100, 0, 5], uncertainty: 10,
        source: 'sparse_observed', status: 'estimated'}]},
    {track_id: 2, class_id: 56, class_name: 'chair', confidence: 0.9, observations: 4,
      frame_id: 30, xyz: [110, 0, 5], position_frame: 30, position_uncertainty: 10,
      position_status: 'estimated', position_source: 'sparse_observed',
      position_history: [{frame_id: 0, timestamp: 0, xyz: [2, 0, 5], uncertainty: 0.1,
        source: 'sparse_observed', status: 'estimated'},
      {frame_id: 30, timestamp: 1, xyz: [110, 0, 5], uncertainty: 10,
        source: 'sparse_observed', status: 'estimated'}]},
  ];
}

test('historical snapshot and relation use historical scatter and frames, never future evidence', () => {
  const app = harness({objects: temporalObjects(), object_observations: [
    {frame_id: 0, timestamp: 0, track_id: 1, class_name: 'bottle', confidence: 0.7,
      xyxy: [40, 30, 60, 50], class_consistency: 1, class_observations: 1},
    {frame_id: 0, timestamp: 0, track_id: 2, class_name: 'chair', confidence: 0.8,
      xyxy: [80, 30, 99, 50], class_consistency: 1, class_observations: 1},
  ]});
  const snapshot = app.run(`currentOnly=true;tNorm=0;positionSnapshot(objs[0])`);
  assert.deepEqual(plain(snapshot.xyz), [0, 0, 5]);
  assert.equal(snapshot.position_frame, 0);
  close(snapshot.position_uncertainty, 0.1);
  assert.equal(snapshot.position_temporal_consistency, 'consistent');
  app.run('updateRelation()');
  const relation = textOf(app.get('relationReadout'));
  assert.match(relation, /右侧/);
  assert.match(relation, /第 0\s*\/\s*0 帧/);
  assert.doesNotMatch(relation, /左右排序待核对/);
});

test('current-frame mode never leaks a future first position into an earlier detection', () => {
  const objects = temporalObjects();
  objects[0].position_history = objects[0].position_history.filter(row => row.frame_id === 30);
  const app = harness({objects, object_observations: [{frame_id: 0, track_id: 1}]});
  assert.equal(app.run('currentOnly=true;tNorm=0;currentPosition(objs[0])'), null);
});

test('classification detail consumes the pipeline evidence schema', () => {
  const app = harness({objects: temporalObjects()});
  app.run('selectedTrackId=1;renderDetails()');
  const details = textOf(app.get('objectDetails'));
  assert.match(details, /80%/);
  assert.match(details, /近 3 次/);
  assert.match(details, /杯子/);
  assert.doesNotMatch(details, /旧数据未记录/);
});

test('video frame overlay uses actual source time, without displaying future sampled boxes', () => {
  const app = harness({video_filename: 'source_video.mp4', object_observations: [
    {frame_id: 30, timestamp: 1, track_id: 1, class_name: 'bottle', xyxy: [0, 0, 20, 20]},
  ]});
  app.get('sourceVideo').currentTime = 0.6;
  assert.equal(app.run('sourceFrame()'), 18);
  app.run('tNorm=timeToNorm(video.currentTime);drawVideoOverlay()');
  const context = app.get('videoOverlay').getContext('2d');
  assert.equal(context.calls.filter(call => call.method === 'strokeRect').length, 0,
    'sampled frame 30 at t=1s must not be drawn over source frame 18 at t=0.6s');
});

test('historical 3D label and category filter agree with the current tracking observation', () => {
  const app = harness({objects: temporalObjects(), object_observations: [
    {frame_id: 0, track_id: 1, class_name: 'cup', confidence: 0.7, hits: 1,
      xyxy: [0, 0, 20, 20], class_consistency: 1, class_observations: 1},
  ]});
  app.run('currentOnly=true;tNorm=0;draw()');
  const labels = app.get('view').getContext('2d').calls
    .filter(call => call.method === 'fillText').map(call => String(call.args[0]));
  assert.ok(labels.some(text => text.includes('杯子 #001')),
    'the 3D marker should use the same historical category as the tracking overlay');
  const visible = app.run(`activeClass='cup';visibleMapObjects().map(o=>o.track_id)`);
  assert.deepEqual(plain(visible), [1]);
});

test('pair focus in current-frame mode fits historical positions rather than final positions', () => {
  const app = harness({objects: temporalObjects(), object_observations: [
    {frame_id: 0, track_id: 1, class_name: 'bottle'},
    {frame_id: 0, track_id: 2, class_name: 'chair'},
  ]});
  app.run('currentOnly=true;tNorm=0');
  app.get('pairBtn').events.click();
  assert.deepEqual(plain(app.run('target')), [1, 0, 5]);
});

test('video HUD names the source frame even between sampled camera poses', () => {
  const app = harness({video_filename: 'source_video.mp4'});
  app.get('sourceVideo').currentTime = 0.6;
  app.run('tNorm=timeToNorm(video.currentTime);updateHud()');
  assert.match(app.get('videoTime').textContent, /帧 18/);
  assert.match(app.get('videoTime').textContent, /0\.6 s/);
});

test('source video time beyond the reconstructed interval stays truthful in the video HUD', () => {
  const app = harness({video_filename: 'source_video.mp4'});
  app.get('sourceVideo').currentTime = 2;
  app.run('tNorm=timeToNorm(video.currentTime);updateHud()');
  assert.match(app.get('videoTime').textContent, /帧 60/);
  assert.match(app.get('videoTime').textContent, /2\.0 s/);
  assert.equal(app.run('currentFrame()'), 30, 'camera pose remains at the last processed frame');
});

test('current classification remains available before the first supported 3D position', () => {
  const objects = temporalObjects();
  objects[0].position_history = objects[0].position_history.filter(row => row.frame_id === 30);
  const app = harness({objects, object_observations: [
    {frame_id: 0, track_id: 1, class_name: 'cup', confidence: 0.7, hits: 1,
      xyxy: [0, 0, 20, 20], class_consistency: 1, class_observations: 1,
      raw_class_name: 'cup', raw_confidence: 0.7},
  ]});
  const snapshot = app.run('currentOnly=true;tNorm=0;positionSnapshot(objs[0])');
  assert.equal(snapshot.xyz, null);
  assert.equal(snapshot.class_name, 'cup');
  close(snapshot.confidence, 0.7);
  assert.equal(snapshot.observations, 1);
  assert.equal(snapshot.frame_id, 0);
});

test('a historical position outside the current detection suspends distances and pair line', () => {
  const app = harness({objects: temporalObjects(), object_observations: [
    // Bottle geometry projects to (50,40), outside this actual detection box.
    {frame_id: 0, track_id: 1, class_name: 'bottle', xyxy: [0, 0, 20, 20]},
    {frame_id: 0, track_id: 2, class_name: 'chair', xyxy: [80, 30, 99, 50]},
  ]});
  const snapshot = app.run('currentOnly=true;tNorm=0;positionSnapshot(objs[0])');
  assert.equal(snapshot.position_temporal_consistency, 'contradicted');
  assert.equal(snapshot.position_status, 'inconsistent');
  assert.deepEqual(plain(snapshot.position_reprojection.pixel), [50, 40]);
  app.run('updateRelation()');
  assert.match(textOf(app.get('relationReadout')), /矛盾/);
  assert.equal(app.get('relationReadout').children.some(node => node.className === 'metrics'), false,
    'contradictory 3D positions must not produce distance metrics; image ordering remains observable');
  const segments = app.run(`
    const contradictorySegments=[];
    line=(a,b)=>contradictorySegments.push([a,b]);
    drawPair(lookAtBasis([0,0,-10],[0,0,0]),600,400);
    contradictorySegments
  `);
  assert.equal(segments.length, 0, 'contradictory positions must not form a distance line');
});

test('overview also suspends distances for a final object flagged contradictory', () => {
  const objects = temporalObjects();
  objects[0].position_temporal_consistency = 'contradicted';
  objects[0].position_status = 'inconsistent';
  const app = harness({objects});
  app.run('currentOnly=false;updateRelation();draw()');
  assert.match(textOf(app.get('relationReadout')), /暂不显示空间距离/);
  const labels = app.get('view').getContext('2d').calls
    .filter(call => call.method === 'fillText').map(call => String(call.args[0]));
  assert.ok(labels.some(label => label.includes('瓶子 #001') && label.includes('待复核')));
});

test('Brown-Conrady projection agrees with an OpenCV oracle after world-camera rotation', () => {
  const app = harness({
    trajectory: [[2, 3, 4]],
    trajectory_timestamps: [0],
    trajectory_frame_ids: [0],
    trajectory_rotations: [[[0, 0, 1], [0, 1, 0], [-1, 0, 0]]],
    summary: {intrinsics: {
      fx: 100, fy: 80, cx: 50, cy: 40, width: 100, height: 80,
      dist: [0.1, 0.01, 0.02, -0.01, 0.005],
    }},
  });
  // This world point has camera coordinates (1, 0.5, 5). Expected image point
  // is independently generated by cv2.projectPoints, including all five terms.
  const projection = app.run(`reprojectToObservation([7,3.5,3],
    {frame_id:0,xyxy:[70.04,48.11,70.06,48.13]},0)`);
  close(projection.pixel[0], 70.0505125, 1e-6);
  close(projection.pixel[1], 48.120205, 1e-6);
  assert.equal(projection.inside_box, true,
    'using an undistorted or mirrored point would fall outside the narrow detection');
});

test('a point behind the rotated source camera cannot agree with any detection box', () => {
  const app = harness({
    trajectory_rotations: [
      [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
      [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    ],
  });
  // World +Z is behind this camera after a half-turn, despite being in front
  // of the independent orbit viewer.
  const projection = app.run(`reprojectToObservation([0,0,5],
    {frame_id:0,xyxy:[0,0,100,80]},.15)`);
  assert.equal(projection.pixel, null);
  assert.equal(projection.inside_box, false);
});

test('missing exact source-camera pose leaves reprojection unverified rather than borrowing one', () => {
  const app = harness();
  const projection = app.run(`reprojectToObservation([0,0,5],
    {frame_id:15,xyxy:[40,30,60,50]},.15)`);
  assert.equal(projection, null);
});

test('independent submaps switch point clouds and never connect trajectory edges', () => {
  const app = harness({
    trajectory: [[0,0,0],[0,0,0],[0,0,0],[1,0,0]],
    trajectory_timestamps: [0,1,2,3],trajectory_frame_ids:[0,1,2,3],
    trajectory_world_frame_ids:['a','a','b','b'],
    map_points:[[0,0,4],[0,0,8]],map_point_world_frame_ids:['a','b'],
    summary:{video:{fps:1},submaps:[{world_frame_id:'a'},{world_frame_id:'b'}],
      frame_quality:{'0':{pose_valid:true,world_frame_id:'a'},'1':{pose_valid:false,world_frame_id:'a'},
        '2':{pose_valid:true,world_frame_id:'b'},'3':{pose_valid:true,world_frame_id:'b'}}},
  });
  assert.deepEqual(plain(app.run('activePoints()')),[[0,0,4]]);
  app.run('tNorm=2/3;syncSubmapView();updateHud()');
  assert.deepEqual(plain(app.run('activePoints()')),[[0,0,8]]);
  assert.equal(app.run('trajectoryVisible(0)'),false);
  assert.equal(app.run('trajectoryEdgeVisible(2)'),false,'must not connect independently scaled origins');
  assert.equal(app.run('trajectoryEdgeVisible(3)'),true);
  assert.match(app.get('trackingNotice').textContent,/空间段 2 \/ 2/);
  assert.match(app.get('trackingNotice').textContent,/不能跨段/);
  app.run('tNorm=1/3;syncSubmapView();updateHud()');
  assert.match(app.get('poseHud').textContent,/三维定位暂时丢失/);
  assert.equal(app.get('trackingNotice').attributes['data-state'],'missing');
  assert.equal(app.run('trajectoryEdgeVisible(1)'),false,'invalid held poses are not a valid path');
});

test('coverage strip preserves gaps and seeks to real source timestamps', () => {
  const app=harness({summary:{geometry_coverage:{segments:[
    {frames:20,start_seconds:0,end_seconds:.63,pose_valid:true},
    {frames:10,start_seconds:.67,end_seconds:.97,pose_valid:false},
  ]}}});
  const segments=app.get('trackingCoverage').children;
  assert.equal(segments.length,2);
  assert.equal(segments[1].attributes['data-state'],'missing');
  assert.equal(segments[0].style.flex,'20');
  assert.match(segments[1].attributes['aria-label'],/定位暂缺/);
  segments[1].events.click();
  close(app.run('normToTime(tNorm)'),.67);
});

test('unverified expanded candidates are opt-in and cannot obscure ordinary results by default', () => {
  const app=harness({objects:[
    {track_id:1,class_name:'bottle',xyz:[0,0,5],depth_source:'sparse_observed',frame_id:0,observations:5},
    {track_id:2,class_name:'earbud case',xyz:[1,0,5],depth_source:'sparse_observed',frame_id:0,
      observations:1,classification_candidate_only:true,
      classification_evidence:{source:'open_vocabulary_model',candidate_only:true,needs_review:true}},
  ]});
  assert.deepEqual(plain(app.run('allObjects().map(o=>o.track_id)')),[1]);
  assert.deepEqual(plain(app.run('visibleMapObjects().map(o=>o.track_id)')),[1]);
  assert.ok(!app.get('objectA').options.some(option=>option.value==='2'));
  app.get('showCandidates').checked=true;app.get('showCandidates').events.change();
  assert.deepEqual(plain(app.run('allObjects().map(o=>o.track_id)')),[1,2]);
  assert.match(textOf(app.get('objectList')),/耳机充电盒.*待核对/);
  app.get('showCandidates').checked=false;app.get('showCandidates').events.change();
  assert.deepEqual(plain(app.run('allObjects().map(o=>o.track_id)')),[1]);
  app.get('classFilter').value='earbud case';app.get('classFilter').events.change();
  assert.equal(app.get('showCandidates').checked,true);
  assert.deepEqual(plain(app.run('allObjects().map(o=>o.track_id)')),[2]);
});

test('confirmed LLM names update visible evidence without replacing original classes or geometry', () => {
  const original={track_id:0,class_name:'remote',xyz:[1,2,3],depth_source:'sparse_observed',frame_id:0,observations:4};
  const app=harness({objects:[original]});
  app.run(`llmUI.applyLabels([{id:'review-1',track_id:0,name:'智能手环',active:true,selections:[]}]);selectObject(objs[0]);`);
  assert.match(textOf(app.get('objectList')),/智能手环/);
  assert.match(textOf(app.get('objectDetails')),/原模型类别.*遥控器/);
  assert.equal(app.run('objs[0].class_name'),'remote');
  assert.equal(app.run('objs[0].track_id'),0,'numeric zero remains the existing track');
  assert.deepEqual(plain(app.run('objs[0].xyz')),[1,2,3]);
  assert.equal(app.run('displayedClass(objs[0])'),'智能手环');
  app.run('llmUI.applyLabels([])');
  assert.equal(app.run('displayedClass(objs[0])'),'remote');
  assert.deepEqual(plain(app.run('objs[0].xyz')),[1,2,3]);
});

test('a newly confirmed missed object remains video-only and appears only in its three evidence frames', () => {
  const app=harness({video_filename:'source_video.mp4'});
  const label={id:'review-2',track_id:null,object_id:'manual_review-2',name:'耳机充电盒',active:true,selections:[
    {frame_id:0,xyxy:[10,10,20,30]},{frame_id:30,xyxy:[11,12,21,32]},{frame_id:60,xyxy:[12,13,22,33]},
  ]};
  app.run(`llmUI.applyLabels(${JSON.stringify([label])})`);
  assert.equal(app.run('objs.length'),1);
  assert.equal(app.run('mapObjs.length'),0);
  assert.equal(app.run('finitePoint(objs[0].xyz)'),false);
  assert.equal(app.run('objs[0].position_relation_usable'),false);
  assert.equal(app.run('occupancyFor(objs[0])'),null);
  assert.deepEqual(plain(app.run('[...observationFrames.keys()]')),[0,30,60]);
  assert.equal(app.run('(observationFrames.get(15)||[]).length'),0);
  assert.equal(app.run("$('objectA').options.some(o=>o.value==='manual_review-2')"),false);
  app.run(`llmUI.applyLabels(${JSON.stringify([label])})`);
  assert.equal(app.run('objs.length'),1,'reloading labels never duplicates an object');
  assert.equal(app.run('observationFrames.get(30).length'),1);
  app.run('selectObject(objs[0])');
  assert.match(textOf(app.get('objectDetails')),/尚无三维几何证据/);
  app.run('llmUI.applyLabels([])');
  assert.equal(app.run('objs.length'),0);
  assert.equal(app.run('observationFrames.get(30).length'),0);
});

test('manual crops require three different paused source frames and never submit without an API key', () => {
  const app=harness({video_filename:'source_video.mp4'});
  app.get('llmStart').events.click();
  const overlay=app.get('videoOverlay');
  function draw(){
    app.get('llmCapture').events.click();
    overlay.events.pointerdown({button:0,pointerId:1,clientX:100,clientY:50,preventDefault(){}});
    overlay.events.pointerup({pointerId:1,clientX:200,clientY:150});
  }
  draw();
  assert.match(app.get('llmStatus').textContent,/已选 1\/3/);
  assert.equal(app.get('sourceVideo').paused,true);
  app.get('llmCapture').events.click();
  assert.match(app.get('llmStatus').textContent,/此时刻已经选过/);
  app.get('sourceVideo').currentTime=1;draw();
  app.get('sourceVideo').currentTime=2;draw();
  assert.equal(app.get('llmCapture').disabled,true);
  assert.equal(app.get('llmSubmit').disabled,true,'three crops alone cannot authorize any provider call');
  assert.match(app.get('llmStatus').textContent,/三张图已选好/);
});

test('configured API is idle until explicit send, then confirmation persists a video-only label', async () => {
  const calls=[];let labels=[];
  const settings={base_url:'https://vision.example/v1',endpoint:'https://vision.example/v1/chat/completions',model:'vision-model',has_api_key:true,key_storage:'session'};
  const app=harness({run_id:'test-run',video_filename:'source_video.mp4'},{
    fetch:async(url,options={})=>{
      calls.push({url,...options});let payload;
      if(url.startsWith('/api/llm/context'))payload={token:'local-test-token',run_id:'test-run',settings};
      else if(url.startsWith('/api/llm/labels?'))payload={labels};
      else if(url==='/api/llm/reviews'&&options.method==='POST')payload={id:'review-3',status:'processing'};
      else if(url==='/api/llm/reviews/review-3')payload={id:'review-3',status:'complete',result:{name:'耳机盒',reason:'三张图均显示盒盖',uncertainty:'品牌未知，可能为不同物体',same_object:false,candidates:['耳机充电盒']}};
      else if(url==='/api/llm/reviews/review-3/confirm'){
        const request=JSON.parse(options.body),review=JSON.parse(calls.find(row=>row.url==='/api/llm/reviews').body);
        labels=[{id:'review-3',run_id:'test-run',track_id:null,object_id:'manual_review-3',name:request.name,selections:review.selections,active:true}];payload={label:labels[0]};
      }else throw new Error('Unexpected API request: '+url);
      return {ok:true,json:async()=>payload};
    },
  });
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(calls.filter(row=>row.method==='POST').length,0,'opening a configured result makes no provider or mutation call');
  app.get('llmStart').events.click();
  for(const time of [0,1,2]){
    app.get('sourceVideo').currentTime=time;app.get('llmCapture').events.click();
    app.get('videoOverlay').events.pointerdown({button:0,pointerId:1,clientX:100,clientY:50,preventDefault(){}});
    app.get('videoOverlay').events.pointerup({pointerId:1,clientX:200,clientY:150});
  }
  assert.equal(app.get('llmSubmit').disabled,false);
  assert.equal(calls.filter(row=>row.method==='POST').length,0,'drawing crops alone cannot send them');
  await app.get('llmSubmit').events.click();
  const submission=calls.find(row=>row.url==='/api/llm/reviews');
  assert.equal(submission.headers['X-VisualMap-Token'],'local-test-token');
  const body=JSON.parse(submission.body);
  assert.equal(body.selections.length,3);
  assert.deepEqual(body.selections.map(row=>row.frame_id),[0,30,60]);
  assert.deepEqual(Object.keys(body.selections[0]).sort(),['frame_id','xyxy']);
  assert.equal(app.run('objs.length'),0,'unconfirmed suggestions cannot change the identification results');
  assert.equal(app.get('llmResponse').hidden,false);
  assert.match(app.get('llmReason').textContent,/三张图/);
  assert.equal(app.get('llmSameObjectWarning').hidden,false);
  assert.equal(app.get('llmConfirm').disabled,true);
  await app.get('llmConfirm').events.click();
  assert.equal(calls.filter(row=>row.url.endsWith('/confirm')).length,0,'same_object=false requires independent explicit acknowledgement');
  app.get('llmSameObjectAck').checked=true;app.get('llmSameObjectAck').events.change();
  assert.equal(app.get('llmConfirm').disabled,false);
  app.get('llmConfirmedName').value='耳机充电盒';
  await app.get('llmConfirm').events.click();
  assert.equal(app.run('objs.length'),1);
  assert.equal(app.run('displayedClass(objs[0])'),'耳机充电盒');
  assert.equal(app.run('finitePoint(objs[0].xyz)'),false);
  assert.match(app.get('llmStatus').textContent,/已保存/);
});

test('saving a manual edit cannot cache or resurrect a revoked server-owned LLM name', () => {
  const values = new Map(), sharedWindow = storedWindow(values);
  const data = () => ({objects: [
    {track_id: 72, class_name: 'dining table', xyz: null},
    {track_id: 73, class_name: 'bottle', xyz: null},
  ]});
  const app = harness(data(), {window: sharedWindow});
  app.run("llmUI.applyLabels([{id:'label-72',track_id:72,name:'大模型确认桌子',active:true}]);selectedTrackId=73");
  app.get('classEdit').value = '人工确认瓶子';
  app.get('classSave').events.click();
  const cached = JSON.parse(values.get(app.run('correctionKey')));
  assert.deepEqual(cached.map(row => row.track_id), [73]);
  assert.equal(cached[0].source, 'user_manual_review');
  app.run('llmUI.applyLabels([])');
  assert.equal(app.run('objectName(objs[0])'), '餐桌 #072');
  const reopened = harness(data(), {window: sharedWindow});
  reopened.run('llmUI.applyLabels([])');
  assert.equal(reopened.run('objectName(objs[0])'), '餐桌 #072');
  assert.equal(reopened.run('objectName(objs[1])'), '人工确认瓶子 #073');
});

test('contaminated legacy caches migrate away LLM rows and preserve legitimate old manual names', () => {
  const values = new Map(), sharedWindow = storedWindow(values);
  const data = () => ({objects: [
    {track_id: 1, class_name: 'bottle', xyz: null},
    {track_id: 2, class_name: 'chair', xyz: null},
    {track_id: 3, class_name: 'cup', xyz: null},
  ]});
  const key = harness(data()).run('correctionKey');
  values.set(key, JSON.stringify([
    {track_id: 1, class_name: '应当撤销的名称', source: 'manual_llm_assisted', label_id: 'revoked'},
    {track_id: 2, class_name: '保留人工椅子', source: 'user_manual_review'},
    {track_id: 3, class_name: '兼容旧人工杯子'},
    null,
  ]));
  const app = harness(data(), {window: sharedWindow});
  app.run('llmUI.applyLabels([])');
  assert.equal(app.run('objectName(objs[0])'), '瓶子 #001');
  assert.equal(app.run('objectName(objs[1])'), '保留人工椅子 #002');
  assert.equal(app.run('objectName(objs[2])'), '兼容旧人工杯子 #003');
  assert.deepEqual(JSON.parse(values.get(key)).map(row => row.track_id), [2, 3]);
  const reopened = harness(data(), {window: sharedWindow});
  assert.equal(reopened.run('objectName(objs[0])'), '瓶子 #001');
});

test('a manual edit after an LLM confirmation survives label reload, revocation and reopening', () => {
  const values = new Map(), sharedWindow = storedWindow(values);
  const data = () => ({objects: [{track_id: 1, class_name: 'remote', xyz: null}]});
  const label = [{id: 'confirmed', track_id: 1, name: '智能手环', active: true}];
  const app = harness(data(), {window: sharedWindow});
  app.run(`llmUI.applyLabels(${JSON.stringify(label)});selectedTrackId=1`);
  app.get('classEdit').value = '人工改为遥控器';
  app.get('classSave').events.click();
  app.run(`llmUI.applyLabels(${JSON.stringify(label)})`);
  assert.equal(app.run('objectName(objs[0])'), '人工改为遥控器 #001');
  const reopened = harness(data(), {window: sharedWindow});
  reopened.run(`llmUI.applyLabels(${JSON.stringify(label)})`);
  assert.equal(reopened.run('objectName(objs[0])'), '人工改为遥控器 #001');
  reopened.run('llmUI.applyLabels([])');
  assert.equal(reopened.run('objectName(objs[0])'), '人工改为遥控器 #001');
  reopened.run("llmUI.applyLabels([{id:'new-confirmation',track_id:1,name:'新的确认名称',active:true}])");
  assert.equal(reopened.run('objectName(objs[0])'), '新的确认名称 #001',
    'a subsequent explicit LLM confirmation can supersede the previous manual edit');
});

test('list, relation selectors and LLM target share current-frame names and preserve choices while seeking', () => {
  const final = {source: 'model', raw_model_class: 'refrigerator', needs_review: true};
  const app = harness({video_filename: 'source_video.mp4', objects: [
    {track_id: 72, class_name: 'dining table', xyz: null, frame_id: 30, classification_evidence: final},
    {track_id: 73, class_name: 'bottle', xyz: null},
    {track_id: 74, class_name: 'chair', xyz: null},
  ], object_observations: [
    {frame_id: 0, track_id: 72, class_name: 'dining table', hits: 1,
      classification_needs_review: false, geometry: {xyz: null, status: 'unavailable'}},
    {frame_id: 30, track_id: 72, class_name: 'dining table', hits: 2,
      classification_needs_review: true, classification_evidence: final,
      geometry: {xyz: null, status: 'unavailable'}},
  ]});
  for (const select of ['objectA', 'objectB', 'llmTarget']) {
    assert.equal(optionText(app, select, 72), '餐桌 #072');
  }
  assert.match(textOf(app.get('objectList')), /餐桌 #072/);
  app.get('objectA').value = '72';app.get('objectB').value = '74';app.get('llmTarget').value = '72';
  app.get('sourceVideo').currentTime = 1;app.run('dirty=true;loop(16)');
  const changed = '待核对目标（模型猜测冰箱） #072';
  assert.match(textOf(app.get('objectList')), /待核对目标（模型猜测冰箱） #072/);
  for (const select of ['objectA', 'objectB', 'llmTarget']) assert.equal(optionText(app, select, 72), changed);
  assert.equal(app.get('objectA').value, '72');assert.equal(app.get('objectB').value, '74');
  assert.equal(app.get('llmTarget').value, '72');
  app.run('selectedTrackId=72');app.get('classEdit').value = '人工确认桌子';app.get('classSave').events.click();
  for (const select of ['objectA', 'objectB', 'llmTarget']) assert.equal(optionText(app, select, 72), '人工确认桌子 #072');
  app.run('populatePairs()');
  assert.equal(app.get('objectB').value, '74', 'reloading names must not silently select a different comparison object');
});

test('a missing current observation clears final geometry and projection and identifies past statistics', () => {
  const app = harness({video_filename: 'source_video.mp4', objects: [{track_id: 1,
    class_name: 'refrigerator', frame_id: 30, observations: 999, confidence: .99,
    xyz: [100, 0, 5], position_frame: 30, position_source: 'sparse_projected',
    position_temporal_consistency: 'consistent', position_relation_usable: true,
    position_reprojection: {frame_id: 30, inside_box: true}, evidence: {sample_count: 999},
    geometry: {frame_id: 30}, occupancy: {available: true},
    classification_evidence: {raw_model_class: 'refrigerator', needs_review: true},
  }], object_observations: [{frame_id: 0, track_id: 1, class_name: 'cup', hits: 2,
    confidence: .7, class_consistency: 1, class_observations: 2,
    geometry: {xyz: [0, 0, 5], frame_id: 0, position_frame: 0, temporal_consistency: 'consistent'}},
  {frame_id: 30, track_id: 1, class_name: 'refrigerator', hits: 999, confidence: .99}]});
  app.get('sourceVideo').currentTime = .5;
  const snapshot = app.run('selectedTrackId=1;renderDetails();positionSnapshot(objs[0])');
  assert.equal(snapshot.observation_current, false);
  assert.equal(snapshot.frame_id, 0);assert.equal(snapshot.class_name, 'cup');assert.equal(snapshot.observations, 2);
  assert.equal(snapshot.xyz, null);assert.equal(snapshot.geometry, null);assert.equal(snapshot.occupancy, null);
  assert.equal(snapshot.position_frame, null);assert.equal(snapshot.position_source, 'none');
  assert.equal(snapshot.position_reprojection, null);assert.deepEqual(plain(snapshot.evidence), {});
  assert.equal(snapshot.position_relation_usable, false);assert.equal(snapshot.position_temporal_consistency, 'unverified');
  const details = textOf(app.get('objectDetails'));
  assert.match(details, /源帧 15 未观测到此物体/);assert.match(details, /第 0 帧历史统计/);
  assert.match(details, /杯子/);assert.match(details, /70%/);
  assert.doesNotMatch(details, /投影落在检测框内|999|第 30 帧|冰箱/);
});

test('a frame before the first observation shows no final recognition or projection evidence', () => {
  const app = harness({video_filename: 'source_video.mp4', objects: [{track_id: 1,
    class_name: 'refrigerator', frame_id: 30, observations: 999, confidence: .99,
    xyz: [0, 0, 5], position_temporal_consistency: 'consistent', evidence: {sample_count: 999},
  }], object_observations: [{frame_id: 30, track_id: 1, class_name: 'refrigerator',
    geometry: {xyz: [0, 0, 5], frame_id: 30}}]});
  const snapshot = app.run('selectedTrackId=1;renderDetails();positionSnapshot(objs[0])');
  assert.equal(snapshot.frame_id, null);assert.equal(snapshot.observations, 0);assert.equal(snapshot.confidence, null);
  assert.equal(snapshot.class_name, '未观测目标');
  const details = textOf(app.get('objectDetails'));
  assert.match(details, /此时刻之前没有识别统计/);
  assert.doesNotMatch(details, /投影落在检测框内|999|第 30 帧|冰箱|99%/);
});

test('terminal review errors retain editable crops and questions for an explicit manual retry and draft restore', async () => {
  const values = new Map(), calls = [], sharedWindow = storedWindow(values);
  const data = () => ({run_id: 'retry-run', video_filename: 'source_video.mp4',
    objects: [{track_id: 72, class_name: 'dining table', xyz: null}]});
  const api = reviewApi(calls, {id: 'retry-review', status: 'error', error: '模拟鉴权失败'});
  const app = harness(data(), {window: sharedWindow, fetch: api});
  await settleAsync();selectReviewCrops(app);
  app.get('llmTarget').value = '72';app.get('llmQuestion').value = '请检查桌子';
  await app.get('llmSubmit').events.click();
  assert.match(app.get('llmStatus').textContent, /模拟鉴权失败.*选图已保留/);
  assert.equal(filledCrops(app).length, 3);
  assert.equal(app.get('llmSubmit').disabled, false);assert.equal(app.get('llmTarget').disabled, false);
  assert.equal(app.get('llmQuestion').disabled, false);
  assert.equal(app.get('llmQuestion').value, '请检查桌子');
  const draftKey = 'visualmap:llm-draft:retry-run';
  assert.equal(JSON.parse(values.get(draftKey)).review_id, null);
  const remove = filledCrops(app)[0].children[1].children[1];
  assert.equal(remove.disabled, false);remove.events.click();
  assert.equal(filledCrops(app).length, 2);selectReviewCrops(app, [3]);
  app.get('llmQuestion').value = '补充检查边界';app.get('llmQuestion').events.input();
  const reopened = harness(data(), {window: sharedWindow, fetch: api});
  await settleAsync();
  assert.equal(filledCrops(reopened).length, 3);assert.equal(reopened.get('llmTarget').value, '72');
  assert.equal(reopened.get('llmQuestion').value, '补充检查边界');
  assert.equal(reopened.get('llmSubmit').disabled, false);
  assert.equal(calls.filter(row => row.url === '/api/llm/reviews').length, 1,
    'editing or restoring a failed draft must not resend pictures');
  await reopened.get('llmSubmit').events.click();
  const submissions = calls.filter(row => row.url === '/api/llm/reviews');
  assert.equal(submissions.length, 2);
  assert.deepEqual(JSON.parse(submissions[1].body).selections.map(row => row.frame_id), [30, 60, 90]);
  assert.equal(JSON.parse(submissions[1].body).question, '补充检查边界');
});

test('an immediate terminal submission error releases the review without polling or discarding crops', async () => {
  const calls = [], app = harness({run_id: 'immediate-error', video_filename: 'source_video.mp4'}, {
    fetch: reviewApi(calls, null, {id: 'retry-review', status: 'error', error: '模拟服务设置错误'}),
  });
  await settleAsync();selectReviewCrops(app);await app.get('llmSubmit').events.click();
  assert.equal(filledCrops(app).length, 3);assert.equal(app.get('llmSubmit').disabled, false);
  assert.match(app.get('llmStatus').textContent, /模拟服务设置错误/);
  assert.equal(calls.some(row => row.url === '/api/llm/reviews/retry-review'), false);
});

test('polling connection errors preserve pending review IDs and cannot unlock duplicate submissions', async () => {
  const values = new Map(), calls = [], sharedWindow = storedWindow(values);
  const data = {run_id: 'pending-run', video_filename: 'source_video.mp4'};
  const app = harness(data, {window: sharedWindow, fetch: reviewApi(calls, new Error('模拟临时断线'))});
  await settleAsync();selectReviewCrops(app);await app.get('llmSubmit').events.click();
  assert.match(app.get('llmStatus').textContent, /任务编号已保留/);
  assert.equal(app.get('llmSubmit').disabled, true);assert.equal(app.get('llmTarget').disabled, true);
  assert.equal(app.get('llmQuestion').disabled, true);assert.equal(app.get('llmReset').disabled, true);
  await app.get('llmSubmit').events.click();app.get('llmReset').events.click();
  assert.equal(filledCrops(app).length, 3);
  assert.equal(JSON.parse(values.get('visualmap:llm-draft:pending-run')).review_id, 'retry-review');
  const restoredCalls = [], reopened = harness(data, {window: sharedWindow,
    fetch: reviewApi(restoredCalls, {id: 'retry-review', status: 'error', error: '服务确认复核中断'}),
  });
  await settleAsync();
  assert.equal(filledCrops(reopened).length, 3);assert.equal(reopened.get('llmSubmit').disabled, false);
  assert.match(reopened.get('llmStatus').textContent, /服务确认复核中断/);
  assert.equal(restoredCalls.some(row => row.url === '/api/llm/reviews/retry-review'), true);
  assert.equal(restoredCalls.some(row => row.method === 'POST'), false,
    'restoring a pending task may query its status, never repeat its provider submission');
  assert.equal(calls.filter(row => row.url === '/api/llm/reviews').length, 1);
});

