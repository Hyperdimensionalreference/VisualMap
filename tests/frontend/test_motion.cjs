'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const motion = require('../../src/visualmap/visualization/viewer_motion.js');

test('closed-form springs start at zero, settle, and critical damping does not bounce', () => {
  assert.equal(motion.springStep(-1), 0);
  assert.equal(motion.springStep(0), 0);
  assert.ok(Math.abs(motion.springStep(2) - 1) < 1e-10);
  let last = 0;
  for (let i = 0; i < 200; i++) {
    const value = motion.springStep(i / 100);
    assert.ok(value >= last && value <= 1);
    last = value;
  }
  for (const damping of [0.96, 1.4]) {
    assert.ok(Number.isFinite(motion.springStep(0.2, 17, damping)));
    assert.ok(Math.abs(motion.springStep(3, 17, damping) - 1) < 1e-7);
  }
});

test('multiple target changes preserve value and velocity at a change', () => {
  const first = [{time: 0, value: 10}];
  const both = [...first, {time: 0.15, value: -5}];
  assert.equal(motion.stepSignal(first, 0.15), motion.stepSignal(both, 0.15));
  const dt = 1e-6;
  const before = (motion.stepSignal(first, 0.15) - motion.stepSignal(first, 0.15 - dt)) / dt;
  const after = (motion.stepSignal(both, 0.15 + dt) - motion.stepSignal(both, 0.15)) / dt;
  assert.ok(Math.abs(before - after) < 0.01);
  assert.ok(Math.abs(motion.stepSignal(both, 3) + 5) < 1e-8);
});

test('seek is independent of render order and event insertion order', () => {
  const events = [
    {time: 0, state: 'selection'}, {time: 0.5, state: 'processing'},
    {time: 1.0, state: 'complete'}, {time: 1.5, state: 'playback'},
    {time: 2.0, state: 'command', details: {label: '查找洞洞板'}},
  ];
  const forward = [0, 0.6, 1.1, 1.7, 2.1].map(t => motion.sample(events, t));
  for (const model of forward.reverse()) {
    assert.deepEqual(motion.sample(events, model.time), model);
    assert.deepEqual(motion.sample([...events].reverse(), model.time), model);
  }
});

test('outgoing content exits before incoming content enters', () => {
  const events = [{time: 1, state: 'command', details: {label: '新内容'}}];
  assert.equal(motion.sample(events, 1.03).contentState, 'idle');
  assert.equal(motion.sample(events, 1.07).contentOpacity, 0);
  assert.equal(motion.sample(events, 1.08).contentOpacity, 0);
  assert.equal(motion.sample(events, 1.2).label, '新内容');
  assert.equal(motion.sample(events, 1.2).contentOpacity, 1);
});

test('direct drag follows pointer and release starts at the actual overshoot', () => {
  const drags = [{start: 1, end: 2, samples: [
    {time: 1, value: 0.2}, {time: 1.5, value: 0.8}, {time: 2, value: 1.3},
  ]}];
  assert.ok(Math.abs(motion.dragSignal(drags, 1.25) - 0.5) < 1e-12);
  assert.equal(motion.dragSignal(drags, 2), 1.3);
  assert.ok(motion.dragSignal(drags, 2.1) > 1);
  assert.ok(Math.abs(motion.dragSignal(drags, 4) - 1) < 1e-10);
  assert.equal(motion.dragSignal(drags, 2, 0, 20, true), 1);
});

test('tab and toggle two-edge springs stretch ahead and settle to their width', () => {
  const changes = [{time: 1, value: 120}];
  const moving = motion.elasticEdges(changes, 1.08, 0, 60);
  assert.ok(moving.width > 60);
  const settled = motion.elasticEdges(changes, 3, 0, 60);
  assert.ok(Math.abs(settled.left - 120) < 1e-10);
  assert.ok(Math.abs(settled.width - 60) < 1e-10);
  assert.deepEqual(motion.elasticEdges(changes, 2, 0, 60, true), {left: 120, right: 180, width: 60});
});

test('all states remain finite under rapid reversals', () => {
  const states = Object.keys(motion.STATES);
  const events = states.map((state, i) => ({time: i * 0.06, state}));
  for (let t = 0; t < 3; t += 0.01) {
    const m = motion.sample(events, t);
    for (const key of ['width', 'height', 'radius', 'blur', 'opacity', 'accent']) {
      assert.ok(Number.isFinite(m[key]), key);
    }
    assert.ok(m.width > 0 && m.height > 0 && m.radius > 0);
    assert.ok(m.opacity >= 0 && m.opacity <= 1 && m.accent >= 0 && m.accent <= 1);
  }
});

function surfaceFixture() {
  let time = 0, pending = null, cancelled = 0;
  const listeners = {};
  const doc = {hidden: false,
    addEventListener(name, fn) {listeners[name] = fn;},
    removeEventListener(name) {delete listeners[name];},
    createElement() {return {textContent: '', style: {}};}};
  const svg = {style: {}};
  const content = {style: {}, innerHTML: '', children: [], append(x) {this.children.push(x);},
    querySelector() {return svg;}};
  const surface = {ownerDocument: doc, dataset: {}, append() {}, setAttribute() {},
    style: {setProperty(name, value) {this[name] = value;}}};
  global.requestAnimationFrame = fn => {pending = fn; return 1;};
  global.cancelAnimationFrame = () => {pending = null; cancelled++;};
  return {surface, content, doc, listeners, clock: () => time,
    advance(t) {time = t; if (pending) {const fn = pending; pending = null; fn();}},
    get cancelled() {return cancelled;}};
}

test('mount reuses the supplied surface and cancels rendering when hidden or destroyed', () => {
  const f = surfaceFixture();
  const controller = motion.mount(f.surface, {content: f.content, clock: f.clock});
  controller.emit('processing');
  f.advance(0.3);
  assert.ok(parseFloat(f.surface.style.width) < 80);
  controller.drag('scrub', 'start', 0.2);
  f.advance(0.4);
  controller.drag('scrub', 'move', 0.6);
  f.advance(0.5);
  assert.equal(Number(f.surface.dataset.dragValue), 0.6);
  f.doc.hidden = true;
  f.listeners.visibilitychange();
  assert.ok(f.cancelled > 0);
  assert.equal(controller.drags.at(-1).end, 0.5);
  controller.destroy();
  assert.equal(f.surface.style.opacity, '0');
  assert.deepEqual(f.listeners, {});
});

test('reduced motion presents the newest real action immediately with no blur', () => {
  const f = surfaceFixture();
  const controller = motion.mount(f.surface, {content: f.content, clock: f.clock, reducedMotion: true});
  controller.emit('command', {label: '查找物体'});
  assert.equal(f.surface.style.width, '300px');
  assert.equal(f.content.style.filter, 'blur(0px)');
  assert.equal(f.content.style.opacity, '1');
  controller.destroy();
});

test('explicit finish has one collapse and cannot flash an empty surface later', () => {
  const f = surfaceFixture();
  const controller = motion.mount(f.surface, {content: f.content, clock: f.clock});
  controller.emit('selection');
  f.advance(0.4);
  controller.finish();
  assert.equal(controller.events.filter(e => e.time > 0.4).length, 0);
  f.advance(2);
  assert.equal(f.surface.style.opacity, '0');
  controller.destroy();
});
