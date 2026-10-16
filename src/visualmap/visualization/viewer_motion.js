/* VisualMap motion: one persistent surface; every rendered value is a time function. */
(function (global) {
  'use strict';
  const clamp = (n, a = 0, b = 1) => Math.max(a, Math.min(b, n));
  const mix = (a, b, p) => a + (b - a) * p;
  const smooth = n => { const p = clamp(n); return p * p * (3 - 2 * p); };
  const aliases = {pick: 'selection', play: 'playback', pause: 'playback', idle: 'idle'};
  const STATES = {
    idle: {w: 192, h: 48, r: 24, label: '', icon: 'plus'},
    selection: {w: 192, h: 48, r: 24, label: '选择视频', icon: 'plus'},
    start: {w: 206, h: 52, r: 26, label: '开始识别', icon: 'arrow'},
    processing: {w: 52, h: 52, r: 26, label: '正在识别', icon: 'loader'},
    complete: {w: 52, h: 52, r: 26, label: '识别完成', icon: 'check', accent: 1},
    playback: {w: 244, h: 76, r: 24, label: '播放 / 暂停', icon: 'play'},
    scrub: {w: 260, h: 58, r: 22, label: '定位视频画面', icon: 'scrub'},
    view: {w: 250, h: 64, r: 24, label: '切换视角', icon: 'cube'},
    object: {w: 270, h: 78, r: 24, label: '选中物体', icon: 'target'},
    display: {w: 236, h: 66, r: 23, label: '调整显示', icon: 'slider'},
    command: {w: 300, h: 92, r: 24, label: '命令搜索', icon: 'search'},
    toast: {w: 260, h: 60, r: 24, label: '已完成', icon: 'check', accent: 1},
    island: {w: 260, h: 76, r: 38, label: '空间工作台', icon: 'cube'},
    player: {w: 440, h: 184, r: 32, label: '视频回看', icon: 'play'},
    volume: {w: 440, h: 120, r: 32, label: '显示强度', icon: 'slider'},
    toggle: {w: 260, h: 86, r: 43, label: '显示场景点', icon: 'toggle'},
    tabs: {w: 500, h: 100, r: 32, label: '切换视角', icon: 'cube'},
    chart: {w: 620, h: 360, r: 36, label: '空间观测', icon: 'chart'},
  };

  // Closed-form unit step. Critical damping avoids bounce; a caller may choose
  // near-critical damping for a very small overshoot. No numerical integration.
  function springStep(t, omega = 17, damping = 1) {
    if (t <= 0) return 0;
    if (Math.abs(damping - 1) < 1e-6) return 1 - (1 + omega * t) * Math.exp(-omega * t);
    if (damping < 1) {
      const q = Math.sqrt(1 - damping * damping), wd = omega * q;
      return 1 - Math.exp(-damping * omega * t) *
        (Math.cos(wd * t) + damping / q * Math.sin(wd * t));
    }
    const q = Math.sqrt(damping * damping - 1);
    const a = -omega * (damping - q), b = -omega * (damping + q);
    return 1 + (b * Math.exp(a * t) - a * Math.exp(b * t)) / (a - b);
  }

  function stateSpec(name, details = {}) {
    const state = aliases[name] || name;
    const preset = STATES[state] || STATES.toast;
    return {state, ...preset, ...details};
  }

  // Sum one step response per target change. The target deltas come from the
  // event ledger, never from whatever happened to be rendered last frame.
  function stepSignal(changes, time, initial = 0, omega = 17, reduced = false) {
    let target = initial, value = initial;
    for (const event of changes) {
      const delta = event.value - target;
      if (event.time <= time) value += delta * (reduced ? 1 : springStep(time - event.time, omega));
      target = event.value;
    }
    return value;
  }

  function sample(events, time, options = {}) {
    const reduced = !!options.reducedMotion;
    const base = stateSpec(options.initial || 'idle', options.initialDetails || {});
    const timeline = events.slice().sort((a, b) => a.time - b.time);
    const samples = timeline.map(e => ({time: e.time, ...stateSpec(e.state, e.details)}));
    const value = key => stepSignal(samples.map(e => ({time: e.time, value: e[key] || 0})),
      time, base[key] || 0, 17, reduced);
    const past = samples.filter(e => e.time <= time);
    const current = past.at(-1) || base;
    const previous = past.at(-2) || base;
    const age = past.length ? time - current.time : Infinity;
    // The outgoing text is gone before the incoming text appears.
    const outgoing = !reduced && age < 0.075;
    const opacity = reduced ? 1 : outgoing ? 1 - smooth(age / 0.06) : smooth((age - 0.085) / 0.09);
    const content = outgoing ? previous : current;
    const blur = reduced ? 0 : outgoing ? smooth(age / 0.06) * 3 : (1 - opacity) * 3;
    const visible = current.state !== 'idle' || age < 0.32;
    return {time, state: current.state, contentState: content.state, label: content.label,
      icon: content.icon, width: value('w'), height: value('h'), radius: value('r'),
      accent: clamp(value('accent')), contentOpacity: opacity, blur,
      opacity: visible ? (current.state === 'idle' ? 1 - smooth(age / 0.32) : 1) : 0,
      details: content, age};
  }

  // While held the cursor is the source of truth. Samples describe actual
  // pointer positions; release uses that exact final value as its initial state.
  function dragSignal(drags, time, initial = 0, omega = 20, reduced = false) {
    let value = initial;
    for (const drag of drags) {
      if (time < drag.start) break;
      const samples = drag.samples || [];
      if (!samples.length) continue;
      let cursor = samples[0].value;
      for (let i = 1; i < samples.length; i++) {
        const a = samples[i - 1], b = samples[i];
        if (time < b.time) {
          cursor = mix(a.value, b.value, clamp((time - a.time) / Math.max(1e-9, b.time - a.time)));
          break;
        }
        cursor = b.value;
      }
      if (drag.end == null || time < drag.end) value = cursor;
      else {
        const release = samples.at(-1).value;
        const target = drag.target == null ? clamp(release) : drag.target;
        value = mix(release, target, reduced ? 1 : springStep(time - drag.end, omega));
      }
    }
    return value;
  }

  // The leading edge arrives first. This same paired-spring construction drives
  // tab highlights and toggle knobs, rather than stretching a rigid rectangle.
  function elasticEdges(changes, time, initialLeft, width, reduced = false) {
    const leftEvents = [], rightEvents = [];
    let target = initialLeft;
    for (const e of changes) {
      const direction = e.value >= target ? 1 : -1;
      leftEvents.push({...e, time: e.time + (direction > 0 ? 0.035 : 0)});
      rightEvents.push({time: e.time + (direction < 0 ? 0.035 : 0), value: e.value + width});
      target = e.value;
    }
    const left = stepSignal(leftEvents, time, initialLeft, 18, reduced);
    const right = stepSignal(rightEvents, time, initialLeft + width, 24, reduced);
    return {left, right, width: Math.max(1, right - left)};
  }

  const paths = {
    plus: '<path d="M12 5v14M5 12h14"/>', arrow: '<path d="M5 12h14m-6-6 6 6-6 6"/>',
    check: '<path d="m5 12 4 4L19 6"/>', play: '<path d="m8 5 11 7-11 7Z"/>',
    pause: '<path d="M8 5v14M16 5v14"/>', loader: '<path d="M20 12a8 8 0 1 1-8-8"/>',
    scrub: '<path d="M4 12h16M9 8v8"/>', slider: '<path d="M4 12h16"/><circle cx="15" cy="12" r="3"/>',
    cube: '<path d="m12 3 9 5v8l-9 5-9-5V8Zm0 10v8M3 8l9 5 9-5M7 5l10 6"/>',
    target: '<circle cx="12" cy="12" r="6"/><path d="M12 2v4m0 12v4M2 12h4m12 0h4"/>',
    search: '<circle cx="10" cy="10" r="6"/><path d="m15 15 5 5"/>',
    toggle: '<rect x="3" y="6" width="18" height="12" rx="6"/><circle cx="15" cy="12" r="3"/>',
    chart: '<path d="M4 4v16h16M7 15l4-5 4 2 5-7"/>',
  };
  function iconSvg(name) {
    return '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
      'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
      (paths[name] || paths.check) + '</svg>';
  }

  function mount(surface, options = {}) {
    if (!surface || !surface.ownerDocument) return {emit() {}, drag() {}, seek() {}, finish() {}, destroy() {}};
    const doc = surface.ownerDocument;
    const content = options.content || doc.createElement('div');
    if (!options.content) surface.append(content);
    const clock = options.clock || (() => global.performance?.now ? global.performance.now() / 1000 : Date.now() / 1000);
    const origin = clock();
    const now = () => clock() - origin;
    const events = [], drags = [];
    const media = global.matchMedia?.('(prefers-reduced-motion: reduce)');
    let reduced = options.reducedMotion ?? media?.matches ?? false;
    let frame = 0, dead = false, textKey = '', activeDrag = null;
    surface.setAttribute('aria-hidden', 'true');
    Object.assign(surface.style, {display: 'flex', alignItems: 'center', justifyContent: 'center',
      boxSizing: 'border-box', overflow: 'hidden', pointerEvents: 'none',
      fontFamily: 'Geist, "Segoe UI", "Microsoft YaHei", sans-serif',
      boxShadow: '0 5px 22px #2c29251a', border: '1px solid #ffffff29'});
    Object.assign(content.style, {display: 'flex', alignItems: 'center', justifyContent: 'center',
      gap: '10px', fontSize: '13px', fontWeight: '500', whiteSpace: 'nowrap', padding: '12px'});
    function seek(time) {
      const model = sample(events, time, {reducedMotion: reduced});
      const gray = [45, 43, 40], orange = [184, 90, 61];
      const color = gray.map((n, i) => Math.round(mix(n, orange[i], model.accent)));
      Object.assign(surface.style, {width: model.width + 'px', height: model.height + 'px',
        borderRadius: model.radius + 'px', background: 'rgb(' + color.join(',') + ')',
        color: '#fffdfa', opacity: String(model.opacity)});
      Object.assign(content.style, {opacity: String(model.contentOpacity), filter: 'blur(' + model.blur + 'px)'});
      const icon = model.details.paused ? 'pause' : model.icon;
      const label = model.width < 95 ? '' : model.label;
      const key = icon + '\n' + label;
      if (textKey !== key) {
        content.innerHTML = iconSvg(icon);
        if (label) {
          const text = doc.createElement('span'); text.textContent = label; text.title = label;
          Object.assign(text.style,{minWidth:'0',overflow:'hidden',textOverflow:'ellipsis'});
          content.append(text);
        }
        textKey = key;
      }
      const svg = content.querySelector('svg');
      if (svg) svg.style.transform = icon === 'loader' && !reduced ? 'rotate(' + (time * 240 % 360) + 'deg)' : 'none';
      if (drags.length) {
        const value = dragSignal(drags, time, 0, 20, reduced);
        surface.style.setProperty('--motion-value', String(value));
        surface.dataset.dragValue = String(value);
      }
      return model;
    }
    function tick() {
      frame = 0;
      if (dead || doc.hidden) return;
      const t = now(), model = seek(t);
      const latest = events.at(-1);
      if (!reduced && (activeDrag || model.state === 'processing' ||
          (latest && t < latest.time + 0.9))) frame = global.requestAnimationFrame(tick);
    }
    function wake() {
      if (dead || doc.hidden) return;
      if (!frame) { seek(now()); if (!reduced) frame = global.requestAnimationFrame(tick); }
    }
    function emit(state, details = {}) {
      const t = now();
      // A scheduled collapse is superseded by the next real user action.
      while (events.length && events.at(-1).time > t) events.pop();
      const normalized = aliases[state] || state;
      events.push({time: t, state: normalized, details});
      if (normalized !== 'idle' && normalized !== 'processing' && !details.persist) {
        events.push({time: t + (details.hold ?? 1.65), state: 'idle'});
      }
      wake();
    }
    function drag(kind, phase, value) {
      const t = now(), numeric = Number(value);
      if (!Number.isFinite(numeric)) return;
      if (phase === 'start' || !activeDrag) {
        emit(kind === 'scrub' ? 'scrub' : 'display', {persist: true});
        activeDrag = {kind, start: t, samples: [{time: t, value: numeric}]};
        drags.push(activeDrag);
      } else activeDrag.samples.push({time: t, value: numeric});
      if (phase === 'end' || phase === 'cancel') {
        activeDrag.end = t; activeDrag.target = clamp(numeric); activeDrag = null;
        emit(kind === 'scrub' ? 'scrub' : 'display', {label: kind === 'scrub' ? '画面已定位' : '显示已调整'});
      }
      wake();
    }
    function visibility() {
      if (doc.hidden) {
        if (frame) global.cancelAnimationFrame(frame);
        frame = 0;
        if (activeDrag) {
          activeDrag.end = now(); activeDrag.target = clamp(activeDrag.samples.at(-1).value); activeDrag = null;
        }
      } else wake();
    }
    function preference(event) {
      reduced = !!event.matches;
      if (frame) global.cancelAnimationFrame(frame);
      frame = 0; wake();
    }
    doc.addEventListener('visibilitychange', visibility);
    media?.addEventListener?.('change', preference);
    seek(0);
    return {emit, drag, seek, events, drags,
      finish() {emit('idle');},
      destroy() {
        dead = true;
        if (frame) global.cancelAnimationFrame(frame);
        frame = 0;
        doc.removeEventListener('visibilitychange', visibility);
        media?.removeEventListener?.('change', preference);
        surface.style.opacity = '0';
      }};
  }
  const api = {STATES, springStep, stepSignal, dragSignal, elasticEdges, sample, mount,
    stateSpec, iconSvg, clamp, mix, smooth};
  global.VisualMapMotion = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
