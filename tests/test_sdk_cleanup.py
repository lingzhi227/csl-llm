"""Own escaped-session cleanup must not signal unrelated/reused identities."""
import os,signal,subprocess,sys,time,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from sdk_probe import cleanup_tree,process_identity


@unittest.skipUnless(sys.platform=='linux','Requires Linux /proc identity tracking')
class CleanupTests(unittest.TestCase):
    def test_escaped_group_and_identity_isolation(self):
        child_code='import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print("ready",flush=True);time.sleep(120)'
        parent_code='import subprocess,sys,signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);p=subprocess.Popen([sys.executable,"-c",'+repr(child_code)+'],start_new_session=True,stdout=subprocess.PIPE,text=True);p.stdout.readline();print(p.pid,flush=True);time.sleep(120)'
        parent=subprocess.Popen([sys.executable,'-c',parent_code],stdout=subprocess.PIPE,text=True,start_new_session=True)
        unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'])
        child=int(parent.stdout.readline());known={pid:process_identity(pid) for pid in (parent.pid,child)}
        try:
            self.assertTrue(all(known.values()))
            self.assertEqual(cleanup_tree({unrelated.pid:'wrong-start-time'}),[])
            self.assertIsNone(unrelated.poll())
            sent=cleanup_tree(known)
            self.assertEqual({r['pid'] for r in sent},set(known))
            self.assertIn(int(signal.SIGKILL),{r['signal'] for r in sent})
            parent.wait(timeout=5);self.assertIsNone(unrelated.poll())
            self.assertTrue(not Path(f'/proc/{child}/stat').exists() or Path(f'/proc/{child}/stat').read_text().split(') ')[1].startswith('Z'))
        finally:
            for pid in known:
                if process_identity(pid)==known[pid]:
                    try:os.kill(pid,signal.SIGKILL)
                    except ProcessLookupError:pass
            unrelated.terminate();unrelated.wait(timeout=5);parent.wait(timeout=5);parent.stdout.close()
if __name__=='__main__':unittest.main()
