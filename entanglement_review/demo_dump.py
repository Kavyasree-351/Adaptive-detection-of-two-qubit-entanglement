import json,re,subprocess,sys,numpy as np
sys.path.insert(0,'/home/claude/tests'); from test_project import JS_HARNESS
html=sys.argv[1]
open('/tmp/h.js','w').write(JS_HARNESS)
d=json.loads(re.search(r"const data = (\{.*?\});\n",open(html).read(),re.S).group(1))
cases=[dict(state=i,noise=n,shots=s) for i in range(6) for n in (0,0.1) for s in (0,500)]
out=json.loads(subprocess.run(['node','/tmp/h.js',html,json.dumps(cases)],capture_output=True,text=True,check=True).stdout)
for c,o in zip(cases,out): print(d['demo'][c['state']]['label'][:28].ljust(28),c['noise'],c['shots'],[round(p,3) for p in o['p']],'|',o['status'].split('. Ground')[0][:40],'| GT',o['status'].split('Ground truth: ')[1].split('.')[0])
