"""Exercise each demo state with Node and portable temporary files."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def main():
    html = Path(sys.argv[1]).resolve()
    os.environ['REPO'] = str(html.parent)
    from test_project import JS_HARNESS
    data = json.loads(re.search(r'const data = (\{.*?\});\n',
                               html.read_text(encoding='utf-8'), re.S).group(1))
    cases = [dict(state=i,noise=n,shots=s) for i in range(len(data['demo']))
             for n in (0,.1) for s in (0,500)]
    with tempfile.TemporaryDirectory() as folder:
        harness = Path(folder)/'harness.js'
        harness.write_text(JS_HARNESS,encoding='utf-8')
        result = subprocess.run(['node',str(harness),str(html),json.dumps(cases)],
                                capture_output=True,text=True,check=True)
    for case,output in zip(cases,json.loads(result.stdout)):
        print(data['demo'][case['state']]['label'],case['noise'],case['shots'],
              output['p'],'|',output['status'])


if __name__ == '__main__':
    main()
