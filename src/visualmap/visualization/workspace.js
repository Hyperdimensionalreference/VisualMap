/* Local video workspace. Motion observes actions; native controls own interaction. */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const icons = {
    cube: '<path d="m12 3 9 5v8l-9 5-9-5V8Zm0 10v8M3 8l9 5 9-5M7 5l10 6"/>',
    library: '<rect x="3" y="5" width="14" height="15" rx="2"/><path d="M7 2h13a1 1 0 0 1 1 1v13M7 9h6M7 13h4"/>',
    upload: '<path d="M12 16V3m-5 5 5-5 5 5M4 14v5a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-5"/>',
    shield: '<path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6Zm-4 9 3 3 5-6"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
    arrow: '<path d="M5 12h14m-6-6 6 6-6 6"/>',
    layers: '<path d="m12 3 10 5-10 5L2 8Zm-9 9 9 5 9-5M3 16l9 5 9-5"/>',
    close: '<path d="m6 6 12 12M6 18 18 6"/>',
    scan: '<path d="M8 3H5a2 2 0 0 0-2 2v3m13-5h3a2 2 0 0 1 2 2v3M3 16v3a2 2 0 0 0 2 2h3m8 0h3a2 2 0 0 0 2-2v-3M3 12h18"/>',
    refresh: '<path d="M20 7v5h-5M4 17v-5h5M6 6a8 8 0 0 1 14 6M4 12a8 8 0 0 0 14 6"/>',
    search: '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>',
    sort: '<path d="M4 6h16M4 12h10M4 18h4m10-5v8m-3-3 3 3 3-3"/>',
    play: '<path d="m9 5 10 7-10 7Z"/>',
    video: '<rect x="3" y="5" width="18" height="14" rx="3"/><path d="m10 9 5 3-5 3Z"/>',
    check: '<path d="m5 12 4 4L19 6"/>',
    clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    warning: '<path d="m12 3 10 18H2Zm0 6v5m0 3v.1"/>',
    trash: '<path d="M3 6h18M9 6V3h6v3M5 6l1 14h12l1-14M10 10v6m4-6v6"/>',
    restore: '<path d="M3 10h6M3 10V4m0 6a9 9 0 1 1 1 9M12 8v5l3 2"/>',
    settings: '<path d="M4 7h16M4 17h16M8 4v6m8 4v6"/>',
  };
  function icon(name, className) {
    const span = document.createElement('span');
    if (className) span.className = className;
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    for (const [key, value] of Object.entries({viewBox:'0 0 24 24',fill:'none',stroke:'currentColor','stroke-width':'1.65','stroke-linecap':'round','stroke-linejoin':'round','aria-hidden':'true'})) svg.setAttribute(key, value);
    // Only developer-owned SVG paths enter this markup; user strings use textContent.
    svg.innerHTML = icons[name] || icons.video;
    span.append(svg);
    return span;
  }
  document.querySelectorAll('[data-icon]').forEach(node => node.replaceChildren(icon(node.dataset.icon).firstChild));
  const motion = window.VisualMapMotion?.mount($('motionSurface'), {content:$('motionContent')});
  const labels = {ready:'待识别',processing:'处理中',complete:'已有结果',error:'识别失败',interrupted:'任务中断',unavailable:'原视频不可用'};
  const cardMap = new Map(), processErrors = new Map();
  let items = [], trashedItems = [], filter = 'all', loaded = false, libraryError = '', refreshPromise = null;
  let libraryMutating = false, lastTrashedId = null, trashRenderKey = '';
  let selected = null, previewURL = null, uploadedItemId = null, uploading = false, starting = false;
  let activeJobId = null, currentJob = null, jobEpoch = 0, pollBusy = false, lastStage = '', pollTimer = 0;
  let detailId = null, detailOpener = null, detailScroll = 0, detailResultsKey = '';
  const supported = /\.(mp4|avi|mov|mkv|webm)$/i;
  const collator = new Intl.Collator('zh-CN', {numeric:true,sensitivity:'base'});
  const node = (tag, text, cls) => {const el = document.createElement(tag); if (text != null) el.textContent = String(text); if (cls) el.className = cls; return el;};
  const number = value => typeof value === 'number' && Number.isFinite(value) ? value : null;
  function size(value) {
    const n = number(value); if (n == null) return '大小未知';
    if (n >= 1024 ** 3) return (n / 1024 ** 3).toFixed(1) + ' GB';
    if (n >= 1024 ** 2) return (n / 1024 ** 2).toFixed(1) + ' MB';
    return Math.max(0, Math.round(n / 1024)) + ' KB';
  }
  function duration(value) {
    const n = number(value); if (n == null || n <= 0) return '时长未知';
    const s = Math.floor(n), h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60);
    return (h ? h + ':' + String(m).padStart(2,'0') : String(m)) + ':' + String(s % 60).padStart(2,'0');
  }
  function date(value, compact = false) {
    if (!value) return '日期未知';
    const parsed = new Date(value); if (!Number.isFinite(parsed.getTime())) return '日期未知';
    return parsed.toLocaleDateString('zh-CN', {year:'numeric',month:'2-digit',day:'2-digit'}) + (compact ? '' : ' ' + parsed.toLocaleTimeString('zh-CN', {hour:'2-digit',minute:'2-digit'}));
  }
  function localURL(value) {
    if (typeof value !== 'string' || !value) return null;
    try { const url = new URL(value, location.href); return url.origin === location.origin && /^https?:$/.test(url.protocol) ? url.href : null; } catch (_) {return null;}
  }
  function statusOf(item) {return Object.hasOwn(labels, item.status) ? item.status : 'ready';}
  function resultsOf(item) {return (Array.isArray(item.results) ? item.results : []).filter(result => localURL(result.result_url));}
  function latestResult(item) {return resultsOf(item).slice().sort((a,b) => (Date.parse(b.created_at)||0) - (Date.parse(a.created_at)||0))[0];}
  function badge(el, item) {const status = statusOf(item); el.className = 'status-badge status-' + status; el.textContent = item.status_label || labels[status];}
  async function request(url, options = {}, timeout = 15000) {
    const controller = new AbortController(), timer = setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch(url, {...options,signal:controller.signal,cache:'no-store'});
      let data; try {data = await response.json();} catch (_) {throw new Error('本机服务未返回有效数据，请刷新后重试。');}
      if (!response.ok) throw new Error(data.error || '请求未完成，请稍后重试。');
      return data;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('本机服务响应超时，请确认工作区仍在运行后重试。');
      if (error instanceof TypeError) throw new Error('暂时无法连接本机服务，请确认工作区仍在运行。');
      throw error;
    } finally {clearTimeout(timer);}
  }
  function feedback(title, detail = '', resultURL = null) {
    $('uploadFeedback').hidden = false;
    $('statusText').textContent = title;
    $('progressText').textContent = detail;
    const url = localURL(resultURL);
    $('openResult').hidden = !url;
    if (url) $('openResult').href = url; else $('openResult').removeAttribute('href');
  }
  function controls() {
    $('chooseBtn').disabled = uploading;
    $('uploadBtn').disabled = uploading;
    $('navUpload').disabled = uploading;
    $('clearFileBtn').disabled = uploading;
    $('saveBtn').disabled = libraryMutating || !selected || uploading || starting || !!uploadedItemId;
    $('startBtn').disabled = libraryMutating || !selected || uploading || starting || !!activeJobId;
    $('undoTrashBtn').disabled = libraryMutating;
    const text = uploadedItemId ? '识别已保存的视频' : '保存并识别';
    $('startBtn').replaceChildren(icon('cube'), document.createTextNode(uploading ? '正在保存视频…' : starting ? '正在启动…' : text));
    if (detailId) updateDetail(false);
  }
  function libraryNotice(message, undoId = null) {
    $('libraryNotice').hidden = false;$('libraryNoticeText').textContent = message;
    lastTrashedId = undoId;$('undoTrashBtn').hidden = !undoId;
  }
  async function changeTrash(id, trashed) {
    if (libraryMutating) return;
    libraryMutating = true;renderLibrary();renderTrash();controls();
    try {
      await request('/api/library/' + encodeURIComponent(id) + (trashed ? '/trash' : '/restore'),{method:'POST',headers:{'Content-Type':'application/json','X-VisualMap-Token':VM_TOKEN},body:'{}'});
      jobEpoch++;
      if (trashed && detailId === String(id)) closeDetail();
      if (trashed && uploadedItemId === String(id)) clearSelection();
      // Let an older periodic refresh finish before reading this mutation back.
      if (refreshPromise) await refreshPromise;
      await refreshLibrary();
      libraryNotice(trashed ? '视频已移到回收站，视频文件和已有结果均已保留。' : '视频已恢复到视频库。',trashed ? String(id) : null);
      motion?.emit('toast',{label:trashed ? '已移到回收站，可恢复' : '视频已恢复',hold:1.5});
    } catch (error) {libraryNotice(error.message);}
    finally {libraryMutating = false;renderLibrary();renderTrash();controls();}
  }
  function renderTrash() {
    $('trashCount').textContent = String(trashedItems.length);
    const key = JSON.stringify([trashedItems,libraryMutating]);if (key === trashRenderKey) return;trashRenderKey = key;
    const list = $('trashList');
    if (!trashedItems.length) {list.replaceChildren(node('p','回收站是空的。','trash-empty'));return;}
    list.replaceChildren(...trashedItems.map(item => {
      const row = node('div',null,'trash-entry'), copy = node('div');
      copy.append(node('b',item.name || '未命名视频'),node('small','移入时间 ' + date(item.deleted_at) + ' · ' + resultsOf(item).length + ' 个识别结果已保留'));
      const restore = node('button',null,'secondary');restore.append(icon('restore'),document.createTextNode('恢复视频'));
      restore.disabled = libraryMutating;restore.addEventListener('click',() => changeTrash(String(item.id),false));
      const purge=node('button','彻底删除','secondary danger-button');purge.disabled=libraryMutating;
      purge.addEventListener('click',()=>previewPurge(item));
      const actions=node('div',null,'trash-entry-actions');actions.append(restore,purge);row.append(copy,actions);return row;
    }));
  }
  function state(title, body, kind, action) {
    const host = $('libraryState'); host.hidden = false;
    host.replaceChildren(icon(kind || 'library'), node('b',title), node('p',body));
    if (action) {const button = node('button',action.label,'secondary'); button.addEventListener('click',action.run); host.append(button);}
  }
  function makeCard(item) {
    const card = node('article',null,'video-card'); card.dataset.videoId = String(item.id);
    const cover = node('button',null,'card-preview'); cover.type = 'button';
    const placeholder = node('div',null,'cover-placeholder'); placeholder.append(icon('video'),node('span','本机视频'));
    const img = node('img'); img.alt = ''; img.loading = 'lazy'; img.hidden = true;
    img.addEventListener('error',() => {img.hidden = true;img.style.opacity = '0';placeholder.hidden = false;});
    img.addEventListener('load',() => {img.hidden = false;img.style.opacity = '1';placeholder.hidden = true;});
    const status = node('span',null,'status-badge'), play = icon('play','play-symbol'), time = node('span',null,'duration');
    cover.append(placeholder,img,play,status,time);
    const body = node('div',null,'card-body'), name = node('button',null,'card-name'), meta = node('p',null,'card-meta'), bottom = node('div',null,'card-bottom');
    name.type = 'button';
    const source = node('span'), sourceText = node('span'); source.append(icon('clock'),sourceText);
    const link = node('a'), detail = node('button','查看详情','text-button'); detail.type = 'button';
    const remove = node('button',null,'card-delete text-button');remove.type = 'button';remove.append(icon('trash'),document.createTextNode('删除'));
    remove.addEventListener('click',() => changeTrash(card.dataset.videoId,true));
    link.append(node('span','打开空间'),icon('arrow'));
    bottom.append(source,link,detail,remove); body.append(name,meta,bottom); card.append(cover,body);
    [cover,name,detail].forEach(button => button.addEventListener('click',() => openDetail(card.dataset.videoId,button)));
    const ref = {card,cover,placeholder,img,status,time,name,meta,sourceText,link,detail,remove,thumb:null};
    cardMap.set(String(item.id),ref); return ref;
  }
  function updateCard(item) {
    const ref = cardMap.get(String(item.id)) || makeCard(item), name = item.name || '未命名视频';
    ref.cover.setAttribute('aria-label','预览视频：' + name); ref.name.textContent = name; ref.name.title = name;
    badge(ref.status,item); ref.time.textContent = duration(item.duration_seconds);
    const resolution = item.width && item.height ? item.width + ' × ' + item.height : '分辨率未知';
    ref.meta.textContent = resolution + ' · ' + size(item.size_bytes);
    ref.sourceText.textContent = date(item.created_at,true);
    const thumbnail = localURL(item.thumbnail_url);
    if (thumbnail !== ref.thumb) {
      ref.thumb = thumbnail;ref.img.style.opacity = '0';ref.img.hidden = !thumbnail;ref.placeholder.hidden = false;
      // A lazy image needs layout to trigger loading. Opacity hides undecoded
      // pixels while preserving that layout; display:none would deadlock it.
      if (thumbnail) ref.img.src = thumbnail;else ref.img.removeAttribute('src');
    }
    const result = latestResult(item); ref.link.hidden = !result; ref.detail.hidden = !!result;
    ref.remove.disabled = libraryMutating || starting || item.status === 'processing';
    ref.remove.setAttribute('aria-label','删除视频：' + name + '（可从回收站恢复）');
    ref.remove.title = item.status === 'processing' ? '识别完成后可删除' : '移到回收站，保留视频与已有结果';
    if (result) ref.link.href = localURL(result.result_url); else ref.link.removeAttribute('href');
    return ref.card;
  }
  function renderLibrary() {
    $('totalCount').textContent = loaded ? String(items.length) : '—';
    $('navCount').textContent = loaded ? String(items.length) : '—';
    $('completeCount').textContent = loaded ? String(items.filter(item => resultsOf(item).length).length) : '—';
    $('storageCount').textContent = loaded ? size(items.reduce((sum,item) => sum + Math.max(0,number(item.size_bytes) || 0),0)) : '—';
    if (!loaded) {if (libraryError) state('视频库暂时无法读取',libraryError,'warning',{label:'重新连接',run:() => refreshLibrary(true)}); return;}
    const search = $('searchInput').value.trim().toLocaleLowerCase();
    const visible = items.filter(item => {
      if (search && !String(item.name || '').toLocaleLowerCase().includes(search)) return false;
      if (filter === 'complete') return resultsOf(item).length > 0;
      if (filter === 'issues') return ['error','interrupted','unavailable'].includes(item.status);
      return filter === 'all' || item.status === filter;
    });
    const sort = $('sortSelect').value;
    visible.sort((a,b) => sort === 'name' ? collator.compare(a.name || '',b.name || '') : sort === 'duration' ? (number(b.duration_seconds)||0) - (number(a.duration_seconds)||0) : ((Date.parse(b.created_at)||0) - (Date.parse(a.created_at)||0)) * (sort === 'oldest' ? -1 : 1));
    $('visibleCount').textContent = visible.length + ' 个视频';
    const grid = $('videoGrid'), activeElement = document.activeElement;
    const needed = new Set(visible.map(item => String(item.id)));
    [...grid.children].forEach(card => {if (!needed.has(card.dataset.videoId)) card.remove();});
    visible.forEach((item,index) => {const card = updateCard(item); if (grid.children[index] !== card) grid.insertBefore(card,grid.children[index] || null);});
    if (activeElement !== document.activeElement && activeElement?.isConnected) activeElement.focus({preventScroll:true});
    if (libraryError) state('连接暂时中断',libraryError + ' 已保留上次读取的列表。','warning',{label:'重新连接',run:() => refreshLibrary(true)});
    else if (!items.length) state('你的第一段视频，从这里开始','导入视频后可在这里回看，并随时启动本机识别。','video',{label:'导入视频',run:pickFile});
    else if (!visible.length) state('没有匹配的视频',search ? '换个关键词，或选择其他状态试试。' : '这个状态下还没有视频。','search',{label:'查看全部视频',run:resetFilters});
    else $('libraryState').hidden = true;
  }
  async function refreshLibrary(userAction = false) {
    if (refreshPromise) return refreshPromise;
    $('refreshBtn').disabled = true;
    const requestedJobEpoch = jobEpoch;
    refreshPromise = (async () => {
      try {
        const data = await request('/api/library');
        if (!Array.isArray(data.items)) throw new Error('视频库格式异常，请重新连接。');
        if (requestedJobEpoch !== jobEpoch) {setTimeout(() => refreshLibrary(),0);return;}
        items = data.items.filter(item => item && item.id != null); loaded = true; libraryError = '';
        trashedItems = (Array.isArray(data.trashed_items) ? data.trashed_items : []).filter(item => item && item.id != null);renderTrash();
        const nextJob = data.active_job_id || null;
        if (requestedJobEpoch === jobEpoch) {
          if (nextJob && nextJob !== activeJobId) {activeJobId = nextJob; currentJob = null; lastStage = '';}
          if (!nextJob && activeJobId && !pollBusy) {activeJobId = null; currentJob = null;}
        }
        renderLibrary(); updateJobPanel(); controls();
        if (userAction) motion?.emit('toast',{label:'视频库已更新',hold:1.2});
      } catch (error) {libraryError = error.message;renderLibrary();}
      finally {$('refreshBtn').disabled = false;refreshPromise = null;}
    })();
    return refreshPromise;
  }
  function resetFilters() {
    filter = 'all';$('searchInput').value = '';
    document.querySelectorAll('[data-filter]').forEach(button => {const on = button.dataset.filter === filter;button.classList.toggle('active',on);button.setAttribute('aria-pressed',String(on));});
    renderLibrary();
  }
  function pickFile() {if (uploading) return;$('file').click();}
  function choose(file) {
    if (!file || uploading) return;
    if (!supported.test(file.name)) {feedback('请选择支持的视频格式','支持 MP4、MOV、AVI、MKV 和 WebM。');return;}
    if (file.size <= 0 || file.size > 2 * 1024 ** 3) {feedback('视频大小不符合要求','请选择大于 0 且不超过 2 GB 的视频。');return;}
    clearSelection(false);selected = file;uploadedItemId = null;previewURL = URL.createObjectURL(file);
    $('preview').src = previewURL;$('selectionPanel').hidden = false;$('filename').textContent = file.name;
    $('fileMeta').textContent = size(file.size) + ' · 尚未保存到本机视频库';
    feedback('视频已选择','可仅保存，或保存后直接进行空间识别。');controls();
    motion?.emit('selection',{label:'视频已选择',hold:1.2});
  }
  function clearSelection(resetInput = true) {
    if (uploading) return;
    $('preview').pause();$('preview').removeAttribute('src');$('preview').load();
    if (previewURL) URL.revokeObjectURL(previewURL);
    previewURL = null;selected = null;uploadedItemId = null;$('selectionPanel').hidden = true;
    if (resetInput) {$('file').value = '';$('uploadFeedback').hidden = true;}
    controls();
  }
  async function saveSelected(andProcess) {
    if (libraryMutating || !selected || uploading || starting || (andProcess && activeJobId)) return;
    const file = selected;uploading = true;controls();
    feedback('正在保存视频到本机视频库','保存结束后会显示真实封面与视频信息。');
    try {
      if (!uploadedItemId) {
        const data = await request('/api/library?filename=' + encodeURIComponent(file.name),{method:'POST',headers:{'Content-Type':'application/octet-stream','X-VisualMap-Token':VM_TOKEN},body:file},180000);
        if (!data.item?.id) throw new Error('视频已接收，但未返回库记录。请刷新视频库核对后再试。');
        uploadedItemId = String(data.item.id);
      }
      uploading = false;controls();
      $('fileMeta').textContent = size(file.size) + ' · 已保存在本机视频库';
      feedback('视频已保存','原始视频会保留，可随时回看或重新识别。');
      await refreshLibrary();
      motion?.emit('complete',{label:'已保存到视频库',hold:1.2});
      if (andProcess) await startProcess(uploadedItemId);
    } catch (error) {feedback('保存未完成',error.message);motion?.emit('toast',{label:'请核对后重试',hold:1.2});}
    finally {uploading = false;controls();}
  }
  async function startProcess(id) {
    if (libraryMutating || starting || uploading || activeJobId) return;
    processErrors.delete(String(id));
    starting = true;controls();feedback('正在启动本机空间识别','视频和结果保存在本机，处理进度来自实际视频帧。');
    try {
      const data = await request('/api/library/' + encodeURIComponent(id) + '/process',{method:'POST',headers:{'Content-Type':'application/json','X-VisualMap-Token':VM_TOKEN},body:'{}'});
      if (!data.id) throw new Error('任务未返回编号，请刷新视频库核对。');
      jobEpoch++;activeJobId = data.id;currentJob = {id:data.id,library_id:id,status:'processing',stage:'正在加载本地模型',frame:0,total_frames:null};lastStage = '';
      const item = items.find(row => String(row.id) === String(id));if (item) {item.status = 'processing';item.status_label = '处理中';item.job_id = data.id;}
      renderLibrary();updateJobPanel();motion?.emit('start',{label:'开始识别',hold:1.2});
      await refreshLibrary();
    } catch (error) {processErrors.set(String(id),error.message);feedback('识别未启动',error.message);motion?.emit('toast',{label:'识别未启动',hold:1.2});}
    finally {starting = false;controls();}
  }
  function jobItem() {return items.find(item => item.job_id === activeJobId || String(item.id) === String(currentJob?.library_id));}
  function frameText(job) {
    const frame = number(job?.frame), total = number(job?.total_frames);
    return frame != null && total != null && total > 0 ? '已处理 ' + Math.max(0,frame) + ' / ' + total + ' 帧' : frame != null && frame > 0 ? '已处理 ' + frame + ' 帧 · 总帧数尚未确认' : '正在准备本地模型与视频，等待实际处理进度';
  }
  function updateJobPanel() {
    $('jobPanel').hidden = !activeJobId;
    if (!activeJobId) return;
    const item = jobItem(), job = currentJob || item || {};
    $('jobTitle').textContent = item?.name || '本机识别任务';
    $('jobStage').textContent = job.stage || '正在处理';$('jobProgress').textContent = frameText(job);
    const bar = $('jobProgressBar'), total = number(job.total_frames), frame = number(job.frame);
    if (total != null && total > 0 && frame != null) {bar.max = total;bar.value = Math.max(0,Math.min(total,frame));}
    else bar.removeAttribute('value');
    $('jobView').disabled = !item;
    if (detailId) updateDetail(false);
  }
  async function readJob() {
    const id = activeJobId;if (!id || pollBusy) return;pollBusy = true;
    try {
      const data = await request('/api/jobs/' + encodeURIComponent(id));
      if (activeJobId !== id) return;
      currentJob = data;updateJobPanel();
      if (data.stage !== lastStage) {lastStage = data.stage || ''; /* Keep long-running progress in its native panel, without a permanent overlay. */}
      const terminal = ['complete','error','interrupted'].includes(data.status);
      if (terminal) {
        jobEpoch++;activeJobId = null;currentJob = null;$('jobPanel').hidden = true;
        if (data.status === 'complete') {
          feedback(data.source_complete ? '空间识别完成，结果已保存' : '已处理范围的空间结果已保存',data.source_complete ? '视频已完整读取，可打开三维空间核对。' : data.requested_partial ? '本次按指定范围处理，视频其余部分尚未处理。' : '视频读取范围尚未核实，请在结果中核对。',data.result_url);
          motion?.emit('complete',{label:'空间结果已保存',hold:1.4});
        } else {feedback(data.status === 'interrupted' ? '识别任务已中断' : '识别未完成',data.error || '原视频仍保存在视频库，可重新识别。');motion?.emit('toast',{label:'可重新识别',hold:1.3});}
        await refreshLibrary();controls();
      }
    } catch (error) {
      $('jobProgress').textContent = error.message + ' 正在尝试重新连接。';
      // A server restart forgets in-memory jobs but preserves the library's
      // interrupted state. Reconcile that state rather than polling a lost ID forever.
      pollBusy = false;await refreshLibrary();
    }
    finally {pollBusy = false;}
  }
  function openDetail(id, opener) {
    const item = items.find(row => String(row.id) === String(id));if (!item) return;
    detailId = String(id);detailOpener = opener || document.activeElement;detailScroll = window.scrollY;detailResultsKey = '';
    const video = $('detailVideo'), url = localURL(item.video_url);
    $('codecHint').hidden = true;video.pause();video.removeAttribute('src');video.removeAttribute('poster');
    const poster = localURL(item.thumbnail_url);if (poster) video.poster = poster;
    video.hidden = !url;$('videoUnavailable').hidden = !!url;
    if (url) video.src = url;video.load();
    updateDetail(true);$('videoDialog').showModal();$('closeDialog').focus({preventScroll:true});
    motion?.emit('object',{label:'查看视频详情',hold:1.1});
  }
  function updateDetail(full) {
    const item = items.find(row => String(row.id) === detailId);if (!item) return;
    badge($('detailBadge'),item);$('dialogTitle').textContent = item.name || '未命名视频';
    $('detailSource').textContent = (item.source_label || (item.source === 'upload' ? '上传视频' : item.source === 'result' ? '历史结果' : '项目视频')) + ' · 保存在本机';
    $('detailResolution').textContent = item.width && item.height ? item.width + ' × ' + item.height : '';
    if (full) {
      const details = [['时长',duration(item.duration_seconds)],['文件大小',size(item.size_bytes)],['文件日期',date(item.created_at)],['分辨率',item.width && item.height ? item.width + ' × ' + item.height : '尚未读取']];
      $('detailMetadata').replaceChildren(...details.map(([title,value]) => {const entry = node('div');entry.append(node('dt',title),node('dd',value));return entry;}));
    }
    const processing = item.status === 'processing' || (activeJobId && String(currentJob?.library_id) === detailId);
    $('processBtn').disabled = libraryMutating || starting || uploading || !!activeJobId || !localURL(item.video_url) || item.status === 'unavailable';
    $('deleteVideoBtn').disabled = libraryMutating || starting || processing;
    $('deleteVideoBtn').title = processing ? '识别完成后可删除' : '移到回收站，视频与已有结果仍保留在本机';
    $('processLabel').textContent = starting ? '正在启动…' : processing ? '正在识别' : resultsOf(item).length ? '再次识别，保留已有结果' : ['error','interrupted'].includes(item.status) ? '重新识别' : '开始识别';
    $('detailStatus').textContent = processing ? (currentJob?.stage || item.stage || '正在处理') + ' · ' + frameText(currentJob || item) : processErrors.has(detailId) ? processErrors.get(detailId) : item.error ? '上次任务：' + item.error : item.status === 'unavailable' ? '原视频缺失，不能重新识别；已保存的空间结果仍可回看。' : activeJobId ? '另一个视频正在识别。当前任务完成后可启动这个视频。' : '每次识别单独保存结果，已有记录会保留。';
    const results = resultsOf(item).slice().sort((a,b) => (Date.parse(b.created_at)||0) - (Date.parse(a.created_at)||0));
    $('resultCount').textContent = results.length ? '(' + results.length + ')' : '';
    const key = JSON.stringify(results);
    if (key !== detailResultsKey) {
      detailResultsKey = key;
      if (!results.length) $('resultList').replaceChildren(node('p','还没有空间结果。启动识别后，可在这里回看每次处理记录。','result-list-empty'));
      else $('resultList').replaceChildren(...results.map(result => {
        const link = node('a',null,'result-entry');link.href = localURL(result.result_url);
        const text = node('div');text.append(node('b',result.title || '三维空间结果'));
        const scope = result.source_complete ? '已完整读取' : result.requested_partial ? '指定范围' : '读取范围未核实';
        text.append(node('small',date(result.created_at,true) + ' · ' + scope + (number(result.processed_frames) != null ? ' · ' + result.processed_frames + ' 帧' : '')));
        if (Number.isInteger(result.tracked_frames) && Number.isInteger(result.processed_frames))
          text.append(node('small','相机有效 ' + result.tracked_frames + '/' + result.processed_frames + ' 帧' +
            (result.submap_count > 1 ? ' · ' + result.submap_count + ' 个独立地图段' : '')));
        link.append(icon('cube','result-icon'),text,icon('arrow','result-arrow'));return link;
      }));
    }
  }
  function closeDetail() {if ($('videoDialog').open) $('videoDialog').close();}
  $('videoDialog').addEventListener('close',() => {
    $('detailVideo').pause();$('detailVideo').removeAttribute('src');$('detailVideo').load();detailId = null;
    window.scrollTo(0,detailScroll);if (detailOpener?.isConnected) detailOpener.focus({preventScroll:true});
  });
  $('closeDialog').addEventListener('click',closeDetail);
  $('videoDialog').addEventListener('click',event => {if (event.target !== $('videoDialog')) return;const rect = $('videoDialog').getBoundingClientRect();if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) closeDetail();});
  $('detailVideo').addEventListener('error',() => {if (detailId && $('detailVideo').getAttribute('src')) $('codecHint').hidden = false;});
  $('detailVideo').addEventListener('play',() => motion?.emit('playback',{label:'预览播放',hold:.8}));
  $('processBtn').addEventListener('click',() => {if (detailId) startProcess(detailId);});
  $('deleteVideoBtn').addEventListener('click',() => {if (detailId) changeTrash(detailId,true);});
  $('undoTrashBtn').addEventListener('click',() => {if (lastTrashedId) changeTrash(lastTrashedId,false);});
  function showTrash(show) {$('trashPanel').hidden = !show;$('trashToggle').setAttribute('aria-expanded',String(show));if (show) $('trashPanel').scrollIntoView({behavior:'smooth',block:'nearest'});}
  $('trashToggle').addEventListener('click',() => showTrash($('trashPanel').hidden));$('closeTrash').addEventListener('click',() => showTrash(false));
  $('jobView').addEventListener('click',event => {const item = jobItem();if (item) openDetail(String(item.id),event.currentTarget);});
  [$('chooseBtn'),$('uploadBtn'),$('navUpload')].forEach(button => button.addEventListener('click',pickFile));
  $('file').addEventListener('change',() => choose($('file').files[0]));
  $('drop').addEventListener('click',event => {if (!event.target.closest('button')) pickFile();});
  $('drop').addEventListener('keydown',event => {if (event.target === $('drop') && ['Enter',' '].includes(event.key)) {event.preventDefault();pickFile();}});
  $('drop').addEventListener('dragover',event => {event.preventDefault();if (event.dataTransfer) event.dataTransfer.dropEffect = uploading ? 'none' : 'copy';if (!uploading) $('drop').classList.add('over');});
  $('drop').addEventListener('dragleave',event => {if (!$('drop').contains(event.relatedTarget)) $('drop').classList.remove('over');});
  function dropFiles(event) {
    event.preventDefault();event.stopPropagation();$('drop').classList.remove('over');document.body.classList.remove('file-dragging');
    const files = Array.from(event.dataTransfer?.files || []);
    if (files.length > 1) {feedback('请一次导入一个视频','可以依次保存多段视频。');return;}
    choose(files[0]);
  }
  $('drop').addEventListener('drop',dropFiles);
  document.addEventListener('dragover',event => {if (Array.from(event.dataTransfer?.types || []).includes('Files')) {event.preventDefault();document.body.classList.add('file-dragging');if (event.dataTransfer) event.dataTransfer.dropEffect = uploading ? 'none' : 'copy';}});
  document.addEventListener('dragleave',event => {if (!event.relatedTarget) document.body.classList.remove('file-dragging');});
  document.addEventListener('drop',event => {if (event.dataTransfer?.files?.length) dropFiles(event);});
  $('clearFileBtn').addEventListener('click',() => clearSelection());
  $('saveBtn').addEventListener('click',() => saveSelected(false));$('startBtn').addEventListener('click',() => saveSelected(true));
  $('refreshBtn').addEventListener('click',() => refreshLibrary(true));
  $('searchInput').addEventListener('input',renderLibrary);$('sortSelect').addEventListener('change',renderLibrary);
  document.querySelectorAll('[data-filter]').forEach(button => button.addEventListener('click',() => {filter = button.dataset.filter;document.querySelectorAll('[data-filter]').forEach(tab => {const active = tab === button;tab.classList.toggle('active',active);tab.setAttribute('aria-pressed',String(active));});renderLibrary();}));
  $('openResult').addEventListener('click',() => motion?.emit('toast',{label:'打开三维空间',hold:1}));
  // Secrets stay in the local service, never in browser storage or rendered HTML.
  let llmSettings = null, settingsBusy = false, storagePlan = null, storageBusy = false, purgePlan = null, purgeBusy = false;
  const postJSON = (url,body) => request(url,{method:'POST',headers:{'Content-Type':'application/json','X-VisualMap-Token':VM_TOKEN},body:JSON.stringify(body)},120000);
  function applyLLMSettings(data) {
    llmSettings=data.settings||data;
    $('llmBaseURL').value=llmSettings.base_url||'';$('llmModel').value=llmSettings.model||'';$('llmAPIKey').value='';
    $('llmAPIKey').placeholder=llmSettings.has_api_key?'已配置，留空保留现有密钥':'未配置密钥';
    $('llmKeyStorage').value=llmSettings.key_storage==='protected'?'protected':'session';
    $('llmKeyStatus').textContent=(llmSettings.has_api_key?(llmSettings.key_storage==='protected'?'密钥已由 Windows 加密保存在本机。':'密钥仅在本次服务运行期间可用，服务重启后需重新填写。'):'尚未配置 API Key。')+(llmSettings.storage_notice?' '+llmSettings.storage_notice:'');
    $('llmNavStatus').classList.toggle('configured',Boolean(llmSettings.has_api_key));
    $('llmSettingsClear').disabled=!llmSettings.has_api_key||settingsBusy;
  }
  async function loadLLMSettings() {
    try {applyLLMSettings(await request('/api/llm/settings'));}
    catch(error){$('llmSettingsStatus').textContent=error.message;}
  }
  function settingsTab(storage) {
    $('llmSettingsPane').hidden=storage;$('storageSettingsPane').hidden=!storage;
    $('llmSettingsTab').setAttribute('aria-selected',String(!storage));$('storageSettingsTab').setAttribute('aria-selected',String(storage));
    if(storage)scanStorage();
  }
  $('navLLMSettings').addEventListener('click',()=>{settingsTab(false);$('llmSettingsStatus').textContent='';$('llmSettingsDialog').showModal();loadLLMSettings();});
  $('llmSettingsClose').addEventListener('click',()=>$('llmSettingsDialog').close());
  $('llmSettingsDialog').addEventListener('close',()=>{$('llmAPIKey').value='';});
  $('llmSettingsTab').addEventListener('click',()=>settingsTab(false));$('storageSettingsTab').addEventListener('click',()=>settingsTab(true));
  $('llmSettingsForm').addEventListener('submit',async event=>{
    event.preventDefault();if(settingsBusy)return;settingsBusy=true;$('llmSettingsSave').disabled=true;
    const payload={base_url:$('llmBaseURL').value.trim(),model:$('llmModel').value.trim(),key_storage:$('llmKeyStorage').value};
    if($('llmAPIKey').value.trim())payload.api_key=$('llmAPIKey').value.trim();
    $('llmAPIKey').value='';$('llmSettingsStatus').textContent='正在保存…';
    try {applyLLMSettings(await postJSON('/api/llm/settings',payload));$('llmSettingsStatus').textContent='设置已保存。打开三维结果，选取三张局部图后即可手动辨认。';}
    catch(error){$('llmSettingsStatus').textContent=error.message;}
    finally {settingsBusy=false;$('llmSettingsSave').disabled=false;$('llmSettingsClear').disabled=!llmSettings?.has_api_key;}
  });
  $('llmSettingsClear').addEventListener('click',async()=>{
    if(settingsBusy)return;settingsBusy=true;$('llmSettingsClear').disabled=true;$('llmAPIKey').value='';
    try{applyLLMSettings(await postJSON('/api/llm/settings',{clear_api_key:true}));$('llmSettingsStatus').textContent='API Key 已清除，暂时无法向大模型发送图片。';}
    catch(error){$('llmSettingsStatus').textContent=error.message;}finally{settingsBusy=false;$('llmSettingsClear').disabled=!llmSettings?.has_api_key;}
  });
  async function scanStorage() {
    if(storageBusy)return;storageBusy=true;$('storageScan').disabled=true;$('storageOptimize').disabled=true;$('storageStatus').textContent='正在核对视频与识别结果，期间可以继续浏览视频库…';
    try{
      const data=await request('/api/storage',{},120000);storagePlan=data.plan;const stats=data.stats||{};
      $('storageSummary').replaceChildren(node('strong',size(stats.allocated_bytes_estimate??stats.unique_file_bytes)),node('span','视频与识别结果估计占用 · '+(stats.file_count??'—')+' 个文件'));
      const count=storagePlan?.operation_count||0,saved=storagePlan?.potential_saved_bytes||0;
      $('storagePlan').replaceChildren(node('b',count?'可无损释放 '+size(saved):'目前没有可整理的重复视频'),node('p',count?'将 '+count+' 个字节完全相同的视频副本复用同一份本机数据。原视频、识别质量和结果均保留。':'已检查项目内视频副本，未发现可安全复用的空间。'));
      const rows=(storagePlan?.operations||[]).slice(0,12).map(op=>node('small',(op.target||'重复视频')+' · '+size(op.bytes)));
      $('storagePlan').append(...rows);$('storageOptimize').disabled=!count;$('storageStatus').textContent=(stats.notes||[]).join(' ');
    }catch(error){storagePlan=null;$('storageStatus').textContent=error.message;}
    finally{storageBusy=false;$('storageScan').disabled=false;}
  }
  $('storageScan').addEventListener('click',scanStorage);
  $('storageOptimize').addEventListener('click',async()=>{
    if(storageBusy||!storagePlan?.plan_token)return;storageBusy=true;$('storageOptimize').disabled=true;$('storageScan').disabled=true;$('storageStatus').textContent='正在无损复用重复视频副本…';
    try {const result=await postJSON('/api/storage/optimize',{plan_token:storagePlan.plan_token});storagePlan=null;
      $('storageStatus').textContent='整理完成，释放 '+size(result.saved_bytes)+'。未重新编码视频，识别质量保持不变。';
      $('storagePlan').replaceChildren(node('b','已整理 '+(result.optimized_files||0)+' 个重复视频副本'));
      if(result.after)$('storageSummary').replaceChildren(node('strong',size(result.after.allocated_bytes_estimate??result.after.unique_file_bytes)),node('span','视频与识别结果估计占用'));
    }catch(error){$('storageStatus').textContent=error.message+' 请重新扫描后再试。';}
    finally{storageBusy=false;$('storageScan').disabled=false;}
  });
  async function previewPurge(item) {
    if(purgeBusy||libraryMutating)return;purgePlan=null;$('purgeConfirm').disabled=true;$('purgeStatus').textContent='正在核对将删除的文件…';
    $('purgeSummary').textContent=item.name||'未命名视频';$('purgeFiles').replaceChildren();$('purgeRetained').replaceChildren();$('purgeDialog').showModal();
    try{
      const plan=await request('/api/library/'+encodeURIComponent(item.id)+'/purge-preview');
      if(!$('purgeDialog').open)return;purgePlan=plan;
      $('purgeSummary').textContent='“'+(plan.name||item.name)+'” · '+plan.file_count+' 个项目文件 · '+size(plan.total_bytes);
      $('purgeFiles').replaceChildren(...(plan.files||[]).map(file=>node('div',file.path+' · '+size(file.bytes))));
      const retained=plan.retained_sources||[];
      $('purgeRetained').replaceChildren(...retained.map(source=>node('p','保留外部原视频：'+source.path+'（'+source.reason+'）')));
      $('purgeStatus').textContent='请核对以上文件，再确认彻底删除。';$('purgeConfirm').disabled=false;
    }catch(error){$('purgeStatus').textContent=error.message;}
  }
  function closePurge(){if(!purgeBusy)$('purgeDialog').close();}
  $('purgeCancel').addEventListener('click',closePurge);$('purgeClose').addEventListener('click',closePurge);
  $('purgeDialog').addEventListener('cancel',event=>{if(purgeBusy)event.preventDefault();});
  $('purgeConfirm').addEventListener('click',async()=>{
    if(purgeBusy||!purgePlan?.confirmation_token)return;purgeBusy=true;libraryMutating=true;$('purgeConfirm').disabled=true;$('purgeCancel').disabled=true;$('purgeClose').disabled=true;
    try{await postJSON('/api/library/'+encodeURIComponent(purgePlan.item_id)+'/purge',{confirmation_token:purgePlan.confirmation_token});
      const removed=String(purgePlan.item_id);purgePlan=null;if(lastTrashedId===removed)lastTrashedId=null;
      $('purgeDialog').close();if(refreshPromise)await refreshPromise;await refreshLibrary();libraryNotice('已彻底删除选定视频的项目文件。外部原视频已保留。');
    }catch(error){purgePlan=null;$('purgeStatus').textContent=error.message+' 请取消后重新打开删除预览。';}
    finally{purgeBusy=false;libraryMutating=false;$('purgeCancel').disabled=false;$('purgeClose').disabled=false;renderLibrary();renderTrash();controls();}
  });
  async function poll() {
    if (activeJobId) await readJob();
    else if (!document.hidden) await refreshLibrary();
    pollTimer = setTimeout(poll,activeJobId ? 1000 : 5000);
  }
  window.addEventListener('pagehide',event => {clearTimeout(pollTimer);if (!event.persisted) {if (previewURL) URL.revokeObjectURL(previewURL);motion?.destroy();}});
  window.addEventListener('pageshow',event => {if (event.persisted) {refreshLibrary();clearTimeout(pollTimer);pollTimer = setTimeout(poll,1000);}});
  document.addEventListener('visibilitychange',() => {if (!document.hidden) refreshLibrary();});
  refreshLibrary().finally(() => {pollTimer = setTimeout(poll,1000);});
})();
