const API_BASE = '/predict'

async function getFeatures(){
  // ask backend for selected features by calling /predict with empty body? We'll instead fetch index to get features via a HEADless call
  // The backend exposes used_features in prediction response; we will request with sample payload if needed.
  // Instead, request a health call to get features by sending an empty sample and expecting an error listing required features.
  return fetch('/static/sample-features.json').then(r=>r.json()).catch(()=>{
    return ['agg','col','nb_usagers','nb_vehicules','v1','plan','situ','vma']
  })
}

function buildForm(features){
  const container = document.getElementById('features')
  container.innerHTML = ''
  features.forEach(f=>{
    const div = document.createElement('div')
    div.className = 'field'
    const label = document.createElement('label')
    label.textContent = f
    const input = document.createElement('input')
    input.type = 'number'
    input.step = 'any'
    input.name = f
    input.id = 'f-'+f
    div.appendChild(label)
    div.appendChild(input)
    container.appendChild(div)
  })
}

function showResult(res){
  document.getElementById('prob').textContent = (res.probability||0).toFixed(4)
  document.getElementById('pred-default').textContent = res.prediction_default
  document.getElementById('pred-thr').textContent = res.prediction_threshold
  document.getElementById('thr').textContent = (res.threshold||0).toFixed(2)
}

function fillSample(features){
  // sample values
  const sample = {agg:2, col:1, nb_usagers:2, nb_vehicules:1, v1:50, plan:1, situ:1, vma:50}
  features.forEach(f=>{
    const el = document.getElementById('f-'+f)
    if(el) el.value = sample[f]===undefined?0:sample[f]
  })
}

function resetForm(features){
  features.forEach(f=>{ const el=document.getElementById('f-'+f); if(el) el.value=''; })
  document.getElementById('prob').textContent='—'
  document.getElementById('pred-default').textContent='—'
  document.getElementById('pred-thr').textContent='—'
  document.getElementById('thr').textContent='—'
}

async function init(){
  const features = await getFeatures()
  buildForm(features)
  const featList = document.getElementById('feat-list')
  featList.innerHTML = ''
  features.forEach(f=>{ const li=document.createElement('li'); li.textContent=f; featList.appendChild(li) })

  document.getElementById('predict-form').addEventListener('submit', async (e)=>{
    e.preventDefault()
    const payload = {}
    features.forEach(f=>{ const val=parseFloat(document.getElementById('f-'+f).value); payload[f]=isNaN(val)?null:val })
    const res = await fetch(API_BASE, {method:'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload)})
    const data = await res.json()
    if(res.ok){ showResult(data) } else { alert('Error: '+(data.error||'unknown')) }
  })

  document.getElementById('use-sample').addEventListener('click', ()=>fillSample(features))
  document.getElementById('reset').addEventListener('click', ()=>resetForm(features))
}

init()

