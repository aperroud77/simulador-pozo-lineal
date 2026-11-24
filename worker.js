importScripts('https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js');
const ready=(async()=>{
 const py=await loadPyodide();
 await py.loadPackage(['numpy','scipy']);
 for(const name of ['fisica.py','puente.py']){
  const r=await fetch(new URL(name,self.location.href));
  if(!r.ok)throw Error('No se pudo cargar '+name);
  py.FS.writeFile(name,await r.text());
 }
 await py.runPythonAsync('import puente'); return py;
})();
ready.then(()=>self.postMessage({ready:true}),e=>self.postMessage({fatal:String(e.message||e)}));
self.onmessage=async({data})=>{
 try{const py=await ready;py.globals.set('entrada',JSON.stringify(data.body));
 const result=py.runPython('puente.ejecutar(entrada)');
 self.postMessage({id:data.id,resultado:JSON.parse(result)});
 }catch(e){self.postMessage({id:data.id,error:String(e.message||e)});}
};
