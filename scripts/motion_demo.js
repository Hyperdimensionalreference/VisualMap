/* Deterministic 28-beat choreography. seek(t) is the sole animation renderer. */
(function () {
  'use strict';
  const M = window.VisualMapMotion;
  const metadata = window.MOTION_METADATA || {};
  const duration = Number(metadata.visual_duration || metadata.duration_sec || metadata.duration) || 14;
  const beat = duration / 28;
  const at = n => n * beat;
  const definitions = [
    ['selection', {w: 220, h: 60, r: 30, label: '选择视频'}],
    ['selection', {w: 260, h: 64, r: 30, label: '选择视频'}],
    ['start', {w: 260, h: 64, r: 30, label: '开始识别'}],
    ['processing', {w: 84, h: 84, r: 42}],
    ['processing', {w: 84, h: 84, r: 42}],
    ['complete', {w: 84, h: 84, r: 42}],
    ['island', {}], ['player', {}], ['player', {}], ['player', {paused: true}],
    ['player', {}], ['player', {}], ['volume', {}], ['volume', {}], ['volume', {}],
    ['toggle', {}], ['tabs', {}], ['tabs', {}],
    ['object', {w: 440, h: 162, r: 32}], ['chart', {}], ['chart', {}], ['chart', {}],
    ['command', {w: 174, h: 70, r: 30}], ['command', {w: 540, h: 244, r: 30}],
    ['command', {w: 540, h: 244, r: 30}], ['toast', {w: 380, h: 88, r: 32}],
    ['selection', {w: 220, h: 60, r: 30, label: '选择视频'}],
    ['selection', {w: 220, h: 60, r: 30, label: '选择视频'}],
  ];
  const ledger = [];
  // Repeating event history supplies the settled periodic initial condition.
  // Endpoint modulo makes the last frame exactly the same as the first.
  for (let cycle = -3; cycle <= 0; cycle++) definitions.forEach(([state, details], i) => {
    ledger.push({time: cycle * duration + at(i), state, details});
  });
  const zoomEvents = ledger.map(e => {
    const s = M.stateSpec(e.state, e.details);
    return {time: e.time, value: Math.min(840 / s.w, 650 / s.h)};
  });
  const progressDrags = [{start: at(10), end: at(11), target: 0.84, samples: [
    {time: at(10), value: 0.18}, {time: at(10.25), value: 0.32},
    {time: at(10.75), value: 0.74}, {time: at(11), value: 0.84}]}];
  const volumeDrags = [{start: at(12), end: at(14), samples: [
    {time: at(12), value: 0.34}, {time: at(12.5), value: 0.72},
    {time: at(13), value: 1}, {time: at(13.5), value: 1.27},
    {time: at(14), value: 1.27}]}];
  const tabChanges = [{time: at(16), value: 0}, {time: at(17), value: 160}];
  const toggleChanges = [{time: at(15), value: 64}];
  const pauseChanges = [{time: at(7), value: 0}, {time: at(9), value: 1},
    {time: at(10), value: 0}];
  const surface = document.getElementById('demoSurface');
  const content = document.getElementById('demoContent');
  const camera = document.getElementById('demoCamera');
  const cursor = document.getElementById('demoCursor');
  const cursorRing = document.getElementById('cursorRing');
  const stage = document.getElementById('demoStage');
  const timeline = document.getElementById('demoTimeline');
  const beatLabel = document.getElementById('beatLabel');
  const stateLabel = document.getElementById('stateLabel');
  const audio = document.getElementById('demoAudio');
  const playbackButton = document.getElementById('demoPlay');
  const soundButton = document.getElementById('demoSound');
  const reducedMedia = window.matchMedia('(prefers-reduced-motion: reduce)');
  let running = false, raf = 0, origin = 0, currentTime = 0, withSound = false;
  let interactive = null, pointerDrag = null, tooltipPointer = null;
  const escape = text => String(text).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;',
    '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
  const icon = name => M.iconSvg(name);
  const button = (name, text, glyph) => '<button type="button" class="demo-button" data-action="' + name + '">' +
    (glyph ? icon(glyph) : '') + (text ? '<span>' + escape(text) + '</span>' : '') + '</button>';
  const player = '<div class="player-content"><div class="player-heading"><span>VIDEO REVIEW</span><span id="playerClock">00:00</span></div>' +
    '<div class="player-row"><div class="video-symbol">' + icon('cube') + '</div><div><strong>空间回看</strong><small>视频 · 逐帧核对</small></div>' +
    '<button type="button" class="play-control" data-action="playback" aria-label="播放或暂停"><svg width="28" height="28" viewBox="0 0 24 24" fill="currentColor"><path id="playMorphA"/><path id="playMorphB"/></svg></button></div>' +
    '<div class="slider-track" data-drag="scrub"><div class="slider-fill" id="progressFill"></div><span class="slider-knob" id="progressKnob"></span>' +
    '<input id="progressInput" aria-label="拖动视频进度" type="range" min="0" max="1" step="0.001"></div></div>';
  const views = {
    selection: button('selection', '选择视频', 'plus'),
    start: button('start', '开始识别', 'arrow'),
    processing: '<svg class="loader" width="38" height="38" viewBox="0 0 48 48" fill="none"><circle cx="24" cy="24" r="17" stroke="currentColor" stroke-opacity=".22" stroke-width="2.4"/>' +
      '<path d="M24 7a17 17 0 0 1 17 17" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/></svg>',
    complete: '<svg width="40" height="40" viewBox="0 0 48 48" fill="none"><path id="checkPath" d="m12 25 8 8 17-19" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" pathLength="1"/></svg>',
    island: '<div class="island-row"><span class="status-dot"></span><span>空间工作台</span><span class="island-mark">' + icon('check') + '</span></div>',
    player,
    volume: '<div class="volume-content"><div class="volume-heading"><span>音量</span><span id="volumeLabel">34%</span></div>' +
      '<div class="slider-track" data-drag="volume"><div class="slider-fill" id="volumeFill"></div><span class="slider-knob" id="volumeKnob"></span>' +
      '<input id="volumeInput" aria-label="拖动音量" type="range" min="0" max="1" step="0.001"></div></div>',
    toggle: '<button type="button" class="toggle-control" data-action="toggle" aria-label="显示场景点" aria-pressed="false"><span id="toggleKnob"></span></button>',
    tabs: '<div class="demo-tabs"><span id="tabIndicator"></span><button type="button" data-action="tab" data-tab="0">立体</button>' +
      '<button type="button" data-action="tab" data-tab="1">顶视</button><button type="button" data-action="tab" data-tab="2">正视</button></div>',
    object: '<div class="object-content"><div class="object-symbol">' + icon('target') + '</div><div><small>选中物体</small><strong>洞洞板</strong>' +
      '<span>回到视频，核对空间位置</span></div>' + button('object', '', 'arrow') + '</div>',
    chart: '<div class="chart-content"><div class="chart-title"><div><small>FRAME OBSERVATIONS</small><strong>空间观测</strong></div><span>逐帧</span></div>' +
      '<svg id="chartSvg" viewBox="0 0 552 248" width="552" height="248" aria-label="示意观测趋势"><path d="M20 48H536M20 108H536M20 168H536M20 228H536" stroke="currentColor" opacity=".12" fill="none"/>' +
      '<path id="chartLine" d="M20 211C48 211 62 153 100 163S151 178 190 126 258 144 284 91 354 111 384 62 440 77 470 36 512 50 536 20" stroke="currentColor" stroke-width="2.4" fill="none" pathLength="1"/>' +
      '<circle id="chartDot" cx="384" cy="62" r="5" fill="#b85a3d" stroke="#fffdfa" stroke-width="2"/></svg><div id="chartTooltip">当前帧<br><small>查看观测证据</small></div></div>',
    command: '<div id="commandCompact"><span>⌘</span><span>K</span></div><div class="command-content" id="commandExpanded">' +
      '<label>' + icon('search') + '<input id="commandInput" autocomplete="off" aria-label="搜索命令" placeholder="查找物体或视角…"><kbd>Esc</kbd></label>' +
      '<div id="commandResults"></div><div class="command-foot"><span>↑ ↓ 选择</span><span>↵ 定位</span></div></div>',
    toast: '<div class="toast-content">' + icon('check') + '<div><strong>已定位洞洞板</strong><small>视角已准备好</small></div></div>',
  };
  const nodes = {};
  Object.entries(views).forEach(([key, html]) => {
    const node = document.createElement('div');
    node.className = 'demo-state demo-' + key;
    node.dataset.state = key;
    node.innerHTML = html;
    node.hidden = true;
    content.append(node);
    nodes[key] = node;
  });
  const labels = ['选择视频', '展开入口', '开始识别', '处理中', '加载反馈', '识别完成', '结果胶囊', '视频回看',
    '播放', '暂停', '拖动时间轴', '松开时间轴', '调整音量', '拖到最大', '回落', '翻转开关',
    '液体指示条', '切换视角', '选中物体', '打开图表', '观测趋势', '悬停提示', '命令入口',
    '输入筛选', '确认定位', '结果提示', '收回入口', '回到起点'];
  const cursorPoints = [
    [500, 706], [596, 703], [760, 701], [826, 782], [860, 743], [815, 744], [967, 721],
    [995, 700], [996, 700], [997, 700], [477, 824], [1038, 823], [526, 735],
    [1046, 735], [1122, 735], [818, 702], [448, 702], [725, 702], [1002, 704],
    [975, 747], [982, 652], [864, 664], [774, 701], [600, 633], [565, 724],
    [928, 731], [811, 702], [588, 706], [500, 706],
  ];
  function cursorSample(t) {
    const i = Math.min(27, Math.floor(t / beat)), p = M.smooth((t - at(i)) / beat);
    const a = cursorPoints[i], b = cursorPoints[i + 1];
    return {x: M.mix(a[0], b[0], p), y: M.mix(a[1], b[1], p)};
  }
  function morphPath(pointsA, pointsB, p) {
    return pointsA.map((point, i) => (i ? 'L' : 'M') +
      M.mix(point[0], pointsB[i][0], p).toFixed(3) + ' ' +
      M.mix(point[1], pointsB[i][1], p).toFixed(3)).join(' ') + 'Z';
  }
  function pureModel(seconds) {
    const t = ((Number(seconds) % duration) + duration) % duration;
    const shape = M.sample(ledger, t, {initial: 'selection', initialDetails: definitions[0][1]});
    let zoom = M.stepSignal(zoomEvents, t, Math.min(840 / 220, 650 / 60), 17);
    const cursorModel = cursorSample(t);
    let volume = M.dragSignal(volumeDrags, t, 0.34);
    let progress = M.dragSignal(progressDrags, t, 0.18);
    const tab = M.elasticEdges(tabChanges, t, 0, 144);
    const toggle = M.elasticEdges(toggleChanges, t, 0, 52);
    const pause = M.stepSignal(pauseChanges, t, 0);
    let chart = M.smooth((t - at(19)) / at(2));
    // Close the actual approach to the seam, not only seek(duration). The final
    // beat removes the tiny infinite spring tail with a smooth, zero-slope
    // blend to the exact initial surface/camera. Hidden controls close too.
    const closure = M.smooth((t - at(27)) / beat);
    if (t < beat) {
      Object.assign(shape, {width: 220, height: 60, radius: 30, accent: 0,
        contentOpacity: 1, blur: 0, opacity: 1});
      zoom = 840 / 220;
    } else if (closure > 0) {
      for (const [key, target] of Object.entries({width: 220, height: 60, radius: 30,
        accent: 0, contentOpacity: 1, blur: 0, opacity: 1})) shape[key] = M.mix(shape[key], target, closure);
      zoom = M.mix(zoom, 840 / 220, closure);
      volume = M.mix(volume, 0.34, closure); progress = M.mix(progress, 0.18, closure);
      chart = M.mix(chart, 0, closure);
      tab.left = M.mix(tab.left, 0, closure); tab.right = M.mix(tab.right, 144, closure); tab.width = tab.right - tab.left;
      toggle.left = M.mix(toggle.left, 0, closure); toggle.right = M.mix(toggle.right, 52, closure); toggle.width = toggle.right - toggle.left;
    }
    zoom = Math.min(zoom, 1190 / shape.width, 870 / shape.height);
    const clickBeats = [1, 2, 5, 6, 7, 8, 9, 15, 16, 17, 18, 22, 24, 26];
    const clickAge = Math.min(...clickBeats.map(b => t >= at(b) ? t - at(b) : Infinity));
    const held = (t >= at(10) && t < at(11)) || (t >= at(12) && t < at(14));
    const click = held ? 1 : clickAge < 0.2 ? 1 - M.smooth(clickAge / 0.2) : 0;
    const typingStart = at(23) + 0.085;
    const typed = t < typingStart || t >= at(25) ? '' : '洞洞板'.slice(0, Math.min(3, Math.floor((t - typingStart) / 0.085) + 1));
    return {t, shape, zoom, cursor: cursorModel, volume, progress, tab, toggle, pause, chart, click, held, typed};
  }
  function seek(seconds, options = {}) {
    const m = pureModel(seconds);
    currentTime = m.t;
    if (!options.interactive) {interactive = null; tooltipPointer = null;}
    const shape = interactive ? M.sample(interactive.events, m.t, {initial: m.shape.state,
      initialDetails: {...m.shape.details, w: m.shape.width, h: m.shape.height, r: m.shape.radius}}) : m.shape;
    const state = shape.contentState;
    const dragVolume = interactive?.volume ?? m.volume;
    const dragProgress = interactive?.progress ?? m.progress;
    const stretch = Math.max(0, dragVolume - 1) * 80;
    surface.style.width = (shape.width + (shape.state === 'volume' ? stretch : 0)) + 'px';
    surface.style.height = shape.height + 'px';
    surface.style.borderRadius = shape.radius + 'px';
    const color = [45, 43, 40].map((n, i) => Math.round(M.mix(n, [184, 90, 61][i], shape.accent)));
    surface.style.background = 'rgb(' + color.join(',') + ')';
    surface.style.color = '#fffdfa';
    camera.style.transform = 'translate(-50%, -50%) scale(' + m.zoom + ')';
    content.style.opacity = String(shape.contentOpacity);
    content.style.filter = 'blur(' + shape.blur + 'px)';
    Object.entries(nodes).forEach(([key, node]) => {node.hidden = key !== state;});
    const loader = nodes.processing.querySelector('.loader');
    loader.style.transform = 'rotate(' + (m.t * 240 % 360) + 'deg)';
    const checkPath = document.getElementById('checkPath');
    checkPath.style.strokeDasharray = '1';
    checkPath.style.strokeDashoffset = String(1 - M.smooth((m.t - at(5)) / 0.24));
    const pause = interactive?.pause ?? m.pause;
    document.getElementById('playMorphA').setAttribute('d', morphPath([[7, 4], [19, 12], [19, 12], [7, 20]], [[7, 4], [10, 4], [10, 20], [7, 20]], pause));
    document.getElementById('playMorphB').setAttribute('d', morphPath([[15, 12], [15, 12], [15, 12], [15, 12]], [[14, 4], [17, 4], [17, 20], [14, 20]], pause));
    const setSlider = (kind, value) => {
      document.getElementById(kind + 'Fill').style.width = M.clamp(value) * 100 + '%';
      document.getElementById(kind + 'Knob').style.left = M.clamp(value) * 100 + '%';
      document.getElementById(kind + 'Input').value = M.clamp(value);
    };
    setSlider('progress', dragProgress); setSlider('volume', dragVolume);
    document.getElementById('volumeLabel').textContent = Math.round(M.clamp(dragVolume) * 100) + '%';
    document.getElementById('playerClock').textContent = '00:' + String(Math.round(M.clamp(dragProgress) * 28)).padStart(2, '0');
    const toggle = interactive?.toggle == null ? m.toggle : M.elasticEdges([{time: interactive.time, value: interactive.toggle}], m.t, m.toggle.left, 52);
    const knob = document.getElementById('toggleKnob');
    knob.style.left = (6 + toggle.left) + 'px'; knob.style.width = toggle.width + 'px';
    nodes.toggle.querySelector('button').setAttribute('aria-pressed', String(toggle.left > 32));
    const tab = interactive?.tab == null ? m.tab : M.elasticEdges([{time: interactive.time, value: interactive.tab * 160}], m.t, m.tab.left, 144);
    document.getElementById('tabIndicator').style.left = (18 + tab.left) + 'px';
    document.getElementById('tabIndicator').style.width = tab.width + 'px';
    nodes.tabs.querySelectorAll('button').forEach((el, i) => {el.style.color = Math.abs(i * 160 - tab.left) < 80 ? '#2d2b28' : '#fffdfa';});
    document.getElementById('chartLine').style.strokeDasharray = '1';
    document.getElementById('chartLine').style.strokeDashoffset = String(1 - m.chart);
    const tip = tooltipPointer || (m.t >= at(21) && m.t < at(22));
    document.getElementById('chartDot').style.opacity = tip ? '1' : '0';
    document.getElementById('chartTooltip').style.opacity = tip ? '1' : '0';
    document.getElementById('chartTooltip').style.filter = 'none';
    const expanded = (m.t >= at(23) + 0.085 && m.t < at(25) + 0.075) || interactive?.command;
    document.getElementById('commandCompact').hidden = !!expanded;
    document.getElementById('commandExpanded').hidden = !expanded;
    const query = interactive?.query ?? m.typed;
    const commandInput = document.getElementById('commandInput');
    if (document.activeElement !== commandInput) commandInput.value = query;
    const results = ['洞洞板', '显示器', '顶视视角'].filter(s => !query || s.includes(query));
    document.getElementById('commandResults').innerHTML = results.map((name, i) => '<button type="button" data-action="result"><span>' + escape(name) + '</span><small>' +
      (name.includes('视角') ? '视角' : '物体') + '</small>' + (i === 0 ? '<kbd>↵</kbd>' : '') + '</button>').join('');
    cursor.style.left = m.cursor.x + 'px'; cursor.style.top = m.cursor.y + 'px';
    cursor.style.transform = 'scale(' + (1 - m.click * 0.13) + ')';
    cursorRing.style.opacity = String(m.click * 0.5);
    cursorRing.style.transform = 'translate(-50%, -50%) scale(' + (1 + (1 - m.click) * 0.3) + ')';
    timeline.value = m.t;
    const i = Math.floor(m.t / beat);
    beatLabel.textContent = String(Math.floor(i / 4) + 1).padStart(2, '0') + ' / ' + (i % 4 + 1);
    stateLabel.textContent = labels[i];
    document.querySelectorAll('.beat-dot').forEach((dot, n) => {
      dot.style.background = n === i ? '#b85a3d' : n < i ? '#2d2b28' : '#d8d4cc';
      dot.style.transform = 'scale(' + (n === i ? 1.35 : 1) + ')';
    });
    return {...m, shape};
  }
  function stop() {
    running = false;
    if (raf) cancelAnimationFrame(raf);
    raf = 0; audio.pause(); playbackButton.textContent = '播放演示';
  }
  function frame(now) {
    if (!running || document.hidden) return;
    const elapsed = (now - origin) / 1000;
    const t = elapsed % duration;
    seek(t);
    if (withSound && audio.src && Math.abs(audio.currentTime - ((Number(metadata.audio_offset) || 0) + t)) > 0.18) {
      audio.currentTime = (Number(metadata.audio_offset) || 0) + t;
    }
    raf = requestAnimationFrame(frame);
  }
  function play() {
    if (reducedMedia.matches) {seek(currentTime); return;}
    running = true; origin = performance.now() - currentTime * 1000;
    playbackButton.textContent = '暂停演示';
    if (withSound && audio.src) {audio.currentTime = (Number(metadata.audio_offset) || 0) + currentTime; audio.play().catch(() => {withSound = false;});}
    if (raf) cancelAnimationFrame(raf);
    raf = requestAnimationFrame(frame);
  }
  function jump(index) {stop(); seek(at(index) + 0.22);}
  content.addEventListener('click', event => {
    const control = event.target.closest('[data-action]');
    if (!control) return;
    stop();
    const action = control.dataset.action;
    if (action === 'selection') jump(2);
    else if (action === 'start') jump(3);
    else if (action === 'object') jump(19);
    else if (action === 'result') jump(25);
    else {
      interactive = interactive || {time: currentTime, events: []};
      if (action === 'playback') {
        interactive.pause = (interactive.pause ?? pureModel(currentTime).pause) > 0.5 ? 0 : 1;
        if (withSound && audio.src && !interactive.pause) audio.play().catch(() => {});
        else audio.pause();
      }
      if (action === 'toggle') interactive.toggle = (interactive.toggle ?? pureModel(currentTime).toggle.left) > 32 ? 0 : 64;
      if (action === 'tab') interactive.tab = Number(control.dataset.tab);
      seek(currentTime + 0.18, {interactive: true});
    }
  });
  content.addEventListener('pointerdown', event => {
    const track = event.target.closest('[data-drag]');
    if (!track || event.button !== 0) return;
    stop(); track.setPointerCapture(event.pointerId);
    pointerDrag = {track, kind: track.dataset.drag};
    updateDrag(event);
    event.preventDefault();
  });
  function updateDrag(event) {
    if (!pointerDrag) return;
    const rect = pointerDrag.track.getBoundingClientRect();
    const value = (event.clientX - rect.left) / Math.max(1, rect.width);
    interactive = interactive || {time: currentTime, events: []};
    interactive[pointerDrag.kind === 'scrub' ? 'progress' : 'volume'] = Math.max(0, value);
    seek(currentTime, {interactive: true});
    if (pointerDrag.kind === 'volume') audio.volume = M.clamp(value);
  }
  content.addEventListener('pointermove', event => {
    if (pointerDrag) updateDrag(event);
    else if (event.target.closest('#chartSvg')) {tooltipPointer = true; seek(currentTime, {interactive: true});}
  });
  function release(event) {
    if (!pointerDrag) return;
    if (!interactive) {pointerDrag = null; return;}
    const key = pointerDrag.kind === 'scrub' ? 'progress' : 'volume';
    const releaseValue = interactive[key], releasedAt = performance.now(), interaction = interactive;
    pointerDrag.track.releasePointerCapture(event.pointerId); pointerDrag = null;
    const renderRelease = now => {
      // Seeking or starting another action retires this release. It must not
      // mutate the replacement interaction or resurrect the previous state.
      if (interactive !== interaction || running || document.hidden) return;
      const age = (now - releasedAt) / 1000;
      interactive[key] = M.mix(releaseValue, M.clamp(releaseValue), M.springStep(age, 20));
      seek(currentTime, {interactive: true});
      if (age < 0.8 && !document.hidden) raf = requestAnimationFrame(renderRelease);
      else raf = 0;
    };
    raf = requestAnimationFrame(renderRelease);
  }
  content.addEventListener('pointerup', release);
  content.addEventListener('pointercancel', release);
  document.getElementById('commandInput').addEventListener('input', event => {
    stop(); interactive = interactive || {time: currentTime, events: []};
    interactive.query = event.target.value; interactive.command = true;
    seek(currentTime, {interactive: true});
  });
  document.addEventListener('keydown', event => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {event.preventDefault(); jump(23); document.getElementById('commandInput').focus();}
    if (event.key === 'Enter' && document.activeElement?.id === 'commandInput') jump(25);
    if (event.key === 'Escape') jump(0);
  });
  playbackButton.addEventListener('click', () => running ? stop() : play());
  soundButton.addEventListener('click', () => {
    if (!audio.src) return;
    withSound = !withSound; soundButton.textContent = withSound ? '音乐已开启' : '开启音乐';
    if (withSound) play(); else audio.pause();
  });
  timeline.addEventListener('input', () => {stop(); seek(Number(timeline.value));});
  document.addEventListener('visibilitychange', () => {if (document.hidden) {stop(); pointerDrag = null;}});
  reducedMedia.addEventListener('change', () => {if (reducedMedia.matches) stop();});
  document.getElementById('restartDemo').addEventListener('click', () => {stop(); seek(0); play();});
  // Public renderer does not dispatch events or advance audio; arbitrary seeks
  // are suitable for beat-frame QA and four-subframe offline capture.
  window.motionDemo = {duration, beat, definitions, labels, ledger, pureModel,
    seek, play, pause: stop, metadata, get currentTime() {return currentTime;}};
  window.seek = seek;
  timeline.max = duration;
  audio.loop = true;
  if (metadata.title) {
    const credit = document.getElementById('musicCredit');
    const link = document.createElement('a');
    link.textContent = metadata.title + ' · ' + (metadata.artist || '') + ' / Mixkit';
    link.href = metadata.source_page || 'https://mixkit.co/free-stock-music/';
    link.target = '_blank'; link.rel = 'noreferrer'; credit.append(link);
  }
  document.getElementById('demoStage').style.aspectRatio = '1 / 1';
  const resize = () => {
    const size = Math.min(window.innerWidth, window.innerHeight);
    stage.style.width = '1440px'; stage.style.height = '1440px';
    stage.style.transform = 'scale(' + size / 1440 + ')';
    document.getElementById('demoFrame').style.width = size + 'px';
    document.getElementById('demoFrame').style.height = size + 'px';
  };
  window.addEventListener('resize', resize); resize(); seek(0);
})();
