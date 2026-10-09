"use strict";
const $ = id => document.getElementById(id);
const state = {map:null, capabilities:null, previousPlan:null, last:null, request:null, imagery:[], vectorIds:[], selected:null};
const fc = features => ({type:"FeatureCollection", features:features || []});
const text = (tag, value, className) => {const el=document.createElement(tag);el.textContent=value ?? "Unknown";if(className)el.className=className;return el;};
const nice = value => String(value || "Unknown").replaceAll("_"," ");
const number = value => Number.isFinite(value) ? value.toFixed(3) : "Not available";
function status(message, error=false){$("status").textContent=message;$("status").className="status"+(error?" error":"");}
async function request(url, body){const response=await fetch(url,body?{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}:{});const data=await response.json();if(!response.ok){const detail=data.detail;throw new Error(typeof detail==="string"?detail:JSON.stringify(detail||"Request failed"));}return data;}
function boundsOf(features){const points=[];function walk(a){if(typeof a?.[0]==="number")points.push(a);else if(Array.isArray(a))a.forEach(walk);}features.forEach(f=>walk(f.geometry?.coordinates));return points.length?points.reduce((b,p)=>[Math.min(b[0],p[0]),Math.min(b[1],p[1]),Math.max(b[2],p[0]),Math.max(b[3],p[1])],[180,90,-180,-90]):null;}
function fit(bounds,padding=60){if(bounds&&state.map)state.map.fitBounds([[bounds[0],bounds[1]],[bounds[2],bounds[3]]],{padding,maxZoom:18,duration:650});}
function source(id, features){state.map?.getSource(id)?.setData(fc(features));}
function addGeometryLayers(sourceId, color, dashed=false){
  const map=state.map;
  map.addSource(sourceId,{type:"geojson",data:fc([])});
  map.addLayer({id:sourceId+"-fill",type:"fill",source:sourceId,filter:["==",["geometry-type"],"Polygon"],paint:{"fill-color":color,"fill-opacity":.2}});
  map.addLayer({id:sourceId+"-line",type:"line",source:sourceId,paint:{"line-color":color,"line-width":2,...(dashed?{"line-dasharray":[2,2]}:{})}});
  map.addLayer({id:sourceId+"-point",type:"circle",source:sourceId,filter:["==",["geometry-type"],"Point"],paint:{"circle-color":color,"circle-radius":5,"circle-stroke-width":1,"circle-stroke-color":"#fff"}});
}
async function loadMap(){
  if(!window.maplibregl)throw new Error("The local map renderer is missing. Run the asset download step.");
  const data=await request("/api/map?tile_limit=64&vector_limit=5000");
  state.mapData=data;
  const attribution=Object.values(data.attribution||{}).filter(v=>typeof v==="string").join(" · ");
  const map=state.map=new maplibregl.Map({container:"map",style:{version:8,sources:{},layers:[{id:"background",type:"background",paint:{"background-color":"#dfe5d7"}}]},center:[0,0],zoom:1,attributionControl:false});
  map.addControl(new maplibregl.NavigationControl({showCompass:false}),"top-right");map.addControl(new maplibregl.ScaleControl({unit:"metric"}),"bottom-left");map.addControl(new maplibregl.AttributionControl({customAttribution:attribution,compact:true}),"bottom-right");
  await new Promise(resolve=>map.on("load",resolve));
  for(const tile of data.tiles){if(!tile.coordinates||tile.coordinates.length!==4)continue;const id="image-"+tile.tile_id;map.addSource(id,{type:"image",url:tile.asset_url,coordinates:tile.coordinates});map.addLayer({id,type:"raster",source:id,paint:{"raster-fade-duration":0}});state.imagery.push(id);}
  const colors={roads:"#efdcb0",buildings:"#f6faf0",drains:"#72b8dd",waterways:"#72b8dd"};
  for(const [name,collection] of Object.entries(data.vector_layers||{})){
    const id="base-"+name;map.addSource(id,{type:"geojson",data:collection});
    map.addLayer({id,type:"line",source:id,paint:{"line-color":colors[name]||"#e9eac4","line-width":name==="buildings"?1:1.6,"line-opacity":.72}});state.vectorIds.push(id);
    const label=document.createElement("label"),input=document.createElement("input");input.type="checkbox";input.checked=true;input.addEventListener("change",()=>map.setLayoutProperty(id,"visibility",input.checked?"visible":"none"));label.append(input,document.createTextNode(" "+nice(name)));$("layers").append(label);
  }
  addGeometryLayers("matches","#75d480");addGeometryLayers("uncertain","#edbd52",true);addGeometryLayers("context","#65bcee");addGeometryLayers("observations","#ffad51",true);addGeometryLayers("selection","#f5f76c");
  for(const id of ["matches-fill","matches-line","uncertain-fill","uncertain-line"]){map.on("click",id,event=>{const id=event.features?.[0]?.id;const all=[...(state.last?.features||[]),...(state.last?.tile_candidates||[]),...(state.last?.uncertain_candidates||[])];const feature=all.find(f=>String(f.id??f.properties?.id??f.properties?.detection_id)===String(id));if(feature)selectFeature(feature);});map.on("mouseenter",id,()=>map.getCanvas().style.cursor="pointer");map.on("mouseleave",id,()=>map.getCanvas().style.cursor="");}
  $("imageryToggle").addEventListener("change",()=>state.imagery.forEach(id=>map.setLayoutProperty(id,"visibility",$("imageryToggle").checked?"visible":"none")));
  $("fitArea").onclick=()=>fit(data.coverage?.bbox_wgs84||boundsOf(data.tiles));
  fit(data.coverage?.bbox_wgs84||boundsOf(data.tiles),50);
  const counts=data.counts?.tiles||{};const truncated=counts.truncated||Object.values(data.counts?.vector_layers||{}).some(c=>c.truncated);
  $("mapNote").textContent=`${data.tiles.length} aerial tiles · ${data.coverage?.acquisition_dates?.join(", ")||"acquisition date unknown"}${truncated?" · map display is limited":""}. Study evidence may differ from conditions today.`;
  if(state.last)renderResults(state.last);
}
async function loadCapabilities(){
  const c=state.capabilities=await request("/api/capabilities");$("area").textContent=c.area_name||"Configured study area";
  const layers=Object.keys(c.vector_layers||{}),reference=layers.includes("roads")?"roads":layers[0];
  for(const [id,concept] of Object.entries(c.supported_concepts||{}).slice(0,3)){
    const label=concept.aliases?.[0]||concept.prompt||nice(id);const button=text("button",reference?`${label} near ${reference}`:label,"example");button.type="button";button.onclick=()=>{$("query").value=reference?`Find ${label} within 100 metres of ${reference}`:`Find ${label}`;$("query").focus();};$("examples").append(button);
  }
  status(`${c.coverage?.tile_count??"?"} prepared tiles. ${c.text_similarity_enabled?"Text-to-image ranking is ready.":"Text-to-image ranking is unavailable; spatial search uses cached detections."}`);
}
function renderResults(data){
  const tiles=data.result_kind==="tile_retrieval",features=tiles?(data.tile_candidates||[]):(data.features||[]),uncertain=data.uncertain_candidates||[];
  const reviewCount=data.total_uncertain??uncertain.length;
  $("resultsTitle").textContent=tiles?`${data.total_tile_candidates??features.length} image candidates`:`${data.total_matches} matches${reviewCount?` · ${reviewCount} need review`:""}`;
  $("scope").textContent=data.claim_scope||"";$("results").replaceChildren();
  if(!features.length&&!uncertain.length){const empty=text("div","No prepared detections satisfy every condition. This does not prove the feature is absent in the real world.","empty");$("results").append(empty);}
  for(const [items,isUncertain] of [[features,false],[uncertain,true]])for(const feature of items){
    const p=feature.properties||{},card=document.createElement("button");card.type="button";card.className="result";card.dataset.id=String(feature.id??p.id??p.detection_id??p.tile_id);
    if(p.asset_url){const img=document.createElement("img");img.src=p.asset_url;img.alt="Aerial evidence tile";img.loading="lazy";card.append(img);}
    const content=text("div","","result-content");content.append(text("h3",isUncertain?"Needs review":tiles?"Image candidate":nice(p.visual_concept||p.concept_id||"Object")));
    const distance=(p.conditions||[]).find(c=>Number.isFinite(c.distance_m))?.distance_m;
    content.append(text("p",`${Number.isFinite(distance)?distance.toFixed(1)+" m to reference · ":""}Imagery ${p.imagery_date||p.acquisition_date||"date unknown"}`));
    content.append(text("span",isUncertain?nice(p.reason||p.evidence_status||"Evidence incomplete"):tiles?"Image similarity; object unverified":"Spatial conditions passed","badge"+(isUncertain?" review":tiles?" tile":"")));
    card.append(content);card.onclick=()=>selectFeature(feature);$("results").append(card);
  }
  source("matches",features);source("uncertain",uncertain);source("context",data.context_features||[]);source("selection",[]);source("observations",[]);$("evidence").hidden=true;
  $("matchLegend").textContent=tiles?"Image candidates":"Matches";
  $("download").hidden=false;$("exportOptions").hidden=!reviewCount;
  const total=tiles?(data.total_tile_candidates??features.length):Math.max(data.total_matches||0,reviewCount),size=state.request.page_size;$("pagination").hidden=total<=size;$("prevPage").disabled=data.page===0;$("nextPage").disabled=(data.page+1)*size>=total;$("pageLabel").textContent=`${data.page+1} / ${Math.max(1,Math.ceil(total/size))}`;
}
function selectFeature(feature){
  const p=feature.properties||{},tile=state.last.result_kind==="tile_retrieval";state.selected=feature;
  document.querySelectorAll(".result").forEach(card=>card.classList.toggle("selected",card.dataset.id===String(feature.id??p.id??p.detection_id??p.tile_id)));
  source("selection",[feature]);source("observations",p.evidence_geometries||[]);fit(boundsOf([feature]),80);
  const root=$("evidenceBody");root.replaceChildren(text("div","Source evidence","eyebrow"),text("h2",tile?"Image candidate":nice(p.visual_concept||p.concept_id||"Object")));
  if(p.asset_url){const frame=text("div","","evidence-image"),img=document.createElement("img");img.src=p.asset_url;img.alt="Source image with the model detection box, when available";frame.append(img);const box=p.pixel_box,w=p.original_width||p.width,h=p.original_height||p.height;if(box?.length===4&&w>0&&h>0){const overlay=text("div","","pixel-box");overlay.style.left=(box[0]/w*100)+"%";overlay.style.top=(box[1]/h*100)+"%";overlay.style.width=((box[2]-box[0])/w*100)+"%";overlay.style.height=((box[3]-box[1])/h*100)+"%";frame.append(overlay);}root.append(frame);}
  const referenceLayer=(p.conditions||[]).find(c=>c.reference_layer)?.reference_layer;
  const vectorDate=p.vector_extraction_date||state.capabilities?.vector_layers?.[referenceLayer]?.extraction_date||"Unknown";
  const dl=document.createElement("dl"),fields=[["Imagery date",p.imagery_date||p.acquisition_date||"Unknown"],["Vector snapshot",tile?"Not used":vectorDate],["Geometry",nice(p.geometry_role||"image tile")],["Detector score",number(p.visual_score)],["Image similarity",number(p.retrieval_similarity)],["Association",nice(p.association?.status||p.association||"Not applicable")],["Source tile",p.tile_id],["Bounds W,S,E,N",boundsOf([feature])?.map(v=>v.toFixed(6)).join(", ")]];
  for(const [label,value] of fields)dl.append(text("dt",label),text("dd",value));root.append(dl);
  root.append(text("p",tile?"Similarity ranks whole images. This candidate has no verified object location or spatial relationship.":"Detector scores are model outputs, not calibrated probabilities. The yellow box shows the original model observation."));
  if(p.conditions?.length){const section=text("div","","conditions");for(const c of p.conditions){const div=text("div","","condition");let explanation=c.reason;if(!explanation){if(c.condition==="visual_detection")explanation=`Model observation of ${nice(c.concept_id)} · score ${number(c.score)}.`;else if(c.condition==="feature_association")explanation=`${nice(c.association)} association with ${nice(c.layer)}${c.feature_id?" · "+c.feature_id:""}.`;else if(c.field)explanation=`${c.field}: observed ${c.observed??"unknown"}; requested ${c.expected}.`;else explanation=`${Number.isFinite(c.distance_m)?c.distance_m.toFixed(2)+" metres to ":""}${nice(c.reference_layer)} on supplied geometry.`;}div.append(text("strong",`${nice(c.condition||c.predicate||"Spatial check")}: ${nice(c.status)}`),text("p",explanation));section.append(div);}root.append(section);}
  const details=document.createElement("details");details.append(text("summary","Full evidence record"),text("pre",JSON.stringify(p,null,2)));root.append(details);$("evidence").hidden=false;
  if(window.matchMedia("(max-width:760px)").matches)$("evidence").scrollIntoView({behavior:"smooth",block:"nearest"});
}
async function search(page=null){
  const query=$("query").value.trim();if(page===null&&query.length<3)return;
  $("searchButton").disabled=true;status("Interpreting the question and checking visual and spatial evidence…");
  try{
    const body=page!==null?{plan:state.last.plan,dataset_version:state.last.dataset_version,mode:state.request.mode,page,page_size:state.request.page_size}:{query,previous_plan:state.previousPlan,mode:document.querySelector('input[name="mode"]:checked').value,page:0,page_size:20};
    const data=await request("/api/query",body);state.last=data;state.request=body;state.previousPlan=data.plan;$("plan").textContent=JSON.stringify(data.plan,null,2);$("planPanel").hidden=false;$("followup").hidden=false;renderResults(data);
    status(`${data.tiles_processed} of ${data.tiles_available} tiles checked. ${data.result_kind==="tile_retrieval"?"Inspect candidates before making object claims.":data.text_similarity_enabled?"Results ranked with the matching image and text model.":"Results use precomputed detections; image ranking is unavailable."}`);
  }catch(error){status(error.message,true);}finally{$("searchButton").disabled=false;}
}
$("searchForm").addEventListener("submit",event=>{event.preventDefault();search();});
$("prevPage").onclick=()=>search(Math.max(0,state.last.page-1));$("nextPage").onclick=()=>search(state.last.page+1);
$("reset").onclick=()=>{state.previousPlan=null;$("followup").hidden=true;status("Follow-up context cleared. Enter a new question.");};
document.querySelectorAll('input[name="mode"]').forEach(input=>input.addEventListener("change",()=>{$("modeHelp").textContent=input.value==="explore"?"Rank image tiles by similarity. Spatial checks need a supported grounded concept.":"Check all prepared object detections against mapped features.";state.previousPlan=null;$("followup").hidden=true;}));
$("closeEvidence").onclick=()=>{$("evidence").hidden=true;source("selection",[]);source("observations",[]);};
$("download").onclick=()=>{if(!state.last)return;const params=new URLSearchParams({plan:JSON.stringify(state.last.plan),mode:state.request.mode,dataset_version:state.last.dataset_version,include_uncertain:String($("includeReview").checked)});const a=document.createElement("a");a.href="/api/export?"+params.toString();a.download="terralens-evidence.geojson";document.body.append(a);a.click();a.remove();status("GeoJSON download requested for all selected results and their supporting evidence.");};
async function start(){const checks=await Promise.allSettled([loadCapabilities(),loadMap()]);checks.forEach(r=>{if(r.status==="rejected"){status(r.reason.message,true);$("mapNote").textContent="A real, prepared evidence bundle is required to display the study area.";}});}
start();
