const input = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const dropText = document.getElementById('dropText');
const btn = document.getElementById('identifyBtn');
const healthBadge = document.getElementById('healthBadge');
const acceptedPill = document.getElementById('acceptedPill');
let selectedFile = null;

fetch('/health').then(r=>r.json()).then(h=>{
  healthBadge.textContent = h.artifacts_ready ? '● model ready' : '○ training required';
  healthBadge.classList.add(h.artifacts_ready ? 'ready':'not-ready');
});

input.addEventListener('change', () => {
  selectedFile = input.files[0];
  if (!selectedFile) return;
  preview.src = URL.createObjectURL(selectedFile);
  preview.hidden = false; dropText.hidden = true; btn.disabled = false;
});

btn.addEventListener('click', async () => {
  if (!selectedFile) return;
  btn.disabled = true; btn.textContent = 'Embedding specimen…';
  const fd = new FormData(); fd.append('file', selectedFile);
  try {
    const res = await fetch('/predict', {method:'POST', body:fd});
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Prediction failed');
    render(data);
  } catch (e) {
    document.getElementById('predictionArea').innerHTML = `<div class="error">${escapeHtml(e.message)}</div>`;
    acceptedPill.textContent = 'no fabricated result';
  } finally {
    btn.disabled = false; btn.textContent = 'Identify specimen';
  }
});

function render(data){
  const accepted = data.accepted || {};
  acceptedPill.textContent = accepted.taxon ? `${accepted.rank}: ${accepted.taxon}` : 'needs morphology';
  const area = document.getElementById('predictionArea');
  area.className='';
  area.innerHTML = ['family','genus','species'].map(rank => {
    const cs = data.predictions[rank] || [];
    if (!cs.length) return '';
    return `<div class="rank"><div class="rank-title"><span>${rank}</span><span>${rank==='genus'?'tap a genus for keys':'model score'}</span></div>${cs.slice(0,4).map(c=>candidate(c,rank)).join('')}</div>`;
  }).join('');

  const family = (data.predictions.family || [])[0]?.taxon;
  area.querySelectorAll('[data-genus]').forEach(row => {
    const select = () => selectGenus(family, row.dataset.genus, area);
    row.addEventListener('click', select);
    row.addEventListener('keydown', event => {
      if(event.key==='Enter' || event.key===' '){ event.preventDefault(); select(); }
    });
  });

  const n = document.getElementById('neighbours'); n.className='thumb-grid';
  n.innerHTML = (data.similar_specimens||[]).map(x=>`<div class="thumb">${x.image_url?`<img loading="lazy" src="${escapeAttr(x.image_url)}" alt="reference specimen">`:''}<div class="meta"><strong>${escapeHtml(x.species||x.genus||x.family||'reference')}</strong>${(x.similarity*100).toFixed(1)}% visual similarity<br>${escapeHtml(x.source||'')} ${x.photo_license?`· ${escapeHtml(x.photo_license)}`:''}</div></div>`).join('') || '<div class="empty">No reference index yet.</div>';

  const d = document.getElementById('diagnostics'); d.className='';
  const chars = data.diagnostics?.characters || [];
  const keys = data.diagnostics?.keys || [];
  d.innerHTML = `${chars.length?`<ul class="chars">${chars.map(x=>`<li>${escapeHtml(x)}</li>`).join('')}</ul>`:'<p class="empty">No family-specific morphology notes yet.</p>'}${keys.length?`<div class="keys"><strong>Keys & references</strong>${keys.map(k=>`<p><em>${escapeHtml(k.title)}</em><br><span class="micro">${escapeHtml(k.type||'')}</span></p>`).join('')}</div>`:''}`;

  renderKeys(data.key_suggestions || {});
  renderKeyGuide(data.key_guide || {});
  const genera = (data.predictions.genus || []).slice(0,4).map(x=>x.taxon).filter(Boolean);
  if (family || genera.length) loadLiveKeys(family, genera);
}
async function selectGenus(family, genus, area){
  area.querySelectorAll('[data-genus]').forEach(row=>row.classList.toggle('selected',row.dataset.genus===genus));
  document.getElementById('keyGuide').innerHTML=`<p class="empty">Building the evidence checklist for <em>${escapeHtml(genus)}</em>…</p>`;
  await Promise.all([loadKeyGuide(family,[genus]),loadLiveKeys(family,[genus])]);
}
async function loadKeyGuide(family,genera){
  const target=document.getElementById('keyGuide');
  const params=new URLSearchParams({family:family||''});
  genera.forEach(genus=>params.append('genera',genus));
  try{
    const response=await fetch(`/keys/navigator?${params}`);
    const payload=await response.json();
    if(!response.ok) throw new Error(payload.detail||'Key navigator failed');
    renderKeyGuide(payload);
  }catch(error){
    target.innerHTML=`<div class="error">Morphology guide unavailable: ${escapeHtml(error.message)}</div>`;
  }
}
function renderKeyGuide(guide){
  const target=document.getElementById('keyGuide');
  const steps=guide.steps||[];
  const candidates=guide.candidate_genera||[];
  if(!steps.length){ target.className='empty'; target.textContent='No structured guide is available for this family yet.'; return; }
  target.className='';
  target.innerHTML=`
    <div class="guide-banner"><strong>${escapeHtml(guide.title||'Morphology checklist')}</strong><span>${escapeHtml(guide.scope||'')}</span></div>
    ${candidates.length?`<div class="candidate-strip"><span>Compare in the key:</span>${candidates.map(x=>`<b>${escapeHtml(x)}</b>`).join('')}</div>`:''}
    <div class="guide-steps">${steps.map((step,index)=>`<label class="guide-step"><input type="checkbox"><span><strong>${escapeHtml(step.title||`${index+1}. Character`)}</strong><em>${escapeHtml(step.inspect||'')}</em><small>${escapeHtml(step.why||'')}</small></span></label>`).join('')}</div>
    ${(guide.stop_rules||[]).length?`<div class="stop-box"><strong>Stop rules</strong><ul>${guide.stop_rules.map(x=>`<li>${escapeHtml(x)}</li>`).join('')}</ul></div>`:''}
    ${(guide.sources||[]).length?`<div class="guide-sources"><strong>Checklist sources</strong>${guide.sources.map(source=>{const href=safeHttpUrl(source.url); const title=escapeHtml(source.title||'Reference'); return `<p>${href?`<a href="${escapeAttr(href)}" target="_blank" rel="noopener noreferrer">${title}</a>`:title}<br><small>${escapeHtml([source.authors,source.year,source.scope].filter(Boolean).join(' · '))}</small></p>`}).join('')}</div>`:''}
    <p class="micro">${escapeHtml(guide.disclaimer||'Checking a box records visible evidence; it does not confirm a genus.')}</p>`;
}
async function loadLiveKeys(family, genera){
  const target = document.getElementById('keyFinder');
  target.className='empty'; target.textContent='Searching curated references, Crossref and OpenAlex…';
  const params = new URLSearchParams({family:family||'', region:'Europe', live:'true'});
  genera.forEach(genus=>params.append('genera',genus));
  try{
    const response = await fetch(`/keys/search?${params}`);
    const payload = await response.json();
    if(!response.ok) throw new Error(payload.detail || 'Key search failed');
    renderKeys(payload);
  }catch(error){
    target.innerHTML=`<div class="error">Live key search unavailable: ${escapeHtml(error.message)}. Curated references remain usable.</div>`;
  }
}
function renderKeys(payload){
  const target = document.getElementById('keyFinder');
  const rows = payload.results || [];
  target.className='';
  if(!rows.length){ target.innerHTML='<p class="empty">No key lead yet. Keep the family candidates and verify morphology first.</p>'; return; }
  target.innerHTML=`<div class="key-list">${rows.map(row=>{
    const title=escapeHtml(row.title||'Untitled resource');
    const href=safeHttpUrl(row.url);
    const heading=href?`<a href="${escapeAttr(href)}" target="_blank" rel="noopener noreferrer">${title}</a>`:`<span>${title}</span>`;
    const meta=[row.authors,row.year,row.type,row.region].filter(Boolean).map(escapeHtml).join(' · ');
    return `<article class="key-hit"><div>${heading}<span class="source-tag">${escapeHtml(row.provider||'reference')}</span></div>${meta?`<p>${meta}</p>`:''}<small>${escapeHtml(row.verification||row.notes||'Check scope before use.')}</small></article>`;
  }).join('')}</div><p class="micro">${escapeHtml(payload.disclaimer||'Candidate resources must be checked by the user.')}</p>`;
}
function candidate(c,rank){ const p=Math.max(0,Math.min(1,c.probability)); const selectable=rank==='genus'; return `<div class="candidate${selectable?' selectable':''}"${selectable?` data-genus="${escapeAttr(c.taxon)}" role="button" tabindex="0" aria-label="Open keys for ${escapeAttr(c.taxon)}"`:''}><div><div class="name">${escapeHtml(c.taxon)}</div><div class="bar"><i style="width:${(p*100).toFixed(1)}%"></i></div></div><div class="score">${(p*100).toFixed(1)}%${selectable?' ↗':''}</div></div>`; }
function safeHttpUrl(value){ try{ const u=new URL(String(value||'')); return ['http:','https:'].includes(u.protocol)?u.href:''; }catch{return '';} }
function escapeHtml(s){return String(s??'').replace(/[&<>'"]/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch]));}
function escapeAttr(s){return escapeHtml(s);}
