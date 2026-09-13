"""PC 端逻辑测试，不加载 KPU，不代表真机识别精度。"""
import sys
import json
import contextlib
import uuid
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules['ujson'] = json
import config
from modules.face_target import FaceTarget
from modules.expression import Expression, features, LABELS
from modules import expression as expression_module
from unittest.mock import patch


def main():
    t = FaceTarget()
    b, changed = t.update([[100,100,100,100]],640,480)
    assert changed and t.snapshot(640,480)['dx'] < 0
    b, changed = t.update([[103,100,100,100],[350,50,200,200]],640,480)
    assert not changed and b[0] == 103  # 已有关联目标优先于更大新目标
    assert t.update([],640,480)[0] is None
    assert t.snapshot(640,480) is None
    b, changed = t.update([[-10,-20,100,100]],640,480)
    assert t.snapshot(640,480)['clipped']
    assert t.update([[0,0,float('nan'),1]],640,480)[0] is None

    p = [v for i in range(106) for v in (220+(i%10)*5,180+(i//10)*5)]
    p[68:70] = [200,200]
    p[176:178] = [300,200]
    f = features(p)
    assert len(f) == 18
    translated = [v+10 for v in p]
    assert all(abs(a-b)<1e-9 for a,b in zip(f,features(translated)))
    assert all(abs(a-b)<1e-9 for a,b in zip(f,features([v*2 for v in p])) )
    workspace = Path(__file__).resolve().parents[1]
    with contextlib.nullcontext(str(workspace / ('test_expression_' + uuid.uuid4().hex))) as d:
        Path(d).mkdir()
        assert Path(d).resolve().is_relative_to(workspace)
        config.EXPRESSION_PROFILE_PATH = d+'/profiles.json'
        config.EXPRESSION_REQUEST_PATH = d+'/request.json'
        now = [1000]
        time.ticks_ms = lambda: now[0]
        time.ticks_diff = lambda a,b: a-b
        e = Expression()
        pose = {'yaw':0,'pitch':0}
        box = [100,100,300,300]
        assert e.update(p,box,pose,1)['reason'] == 'needs_calibration'
        assert e.update(p,[0,0,300,300],pose,1)['reason'] == 'clipped_face'
        assert e.update(p,box,{'yaw':40,'pitch':0},1)['label'] == 'unknown'
        Path(config.EXPRESSION_REQUEST_PATH).write_text(json.dumps({'label':'neutral'}))
        assert e.update(p,box,pose,1)['reason'] == 'calibrating'
        now[0] += 3100
        for _ in range(15): e.update(p,box,pose,1)
        assert 'neutral' in Expression().profiles
        real_rename = expression_module.os.rename
        def fat_rename(src, dst):
            if Path(dst).exists():
                raise OSError(17, 'EEXIST')
            real_rename(src, dst)
        with patch.object(expression_module.os, 'rename', fat_rename):
            for label in ('happy', 'angry', 'sad', 'surprise', 'neutral'):
                updated = dict(Expression().profiles)
                updated[label] = list(f)
                expression_module.save_profiles(updated)
                assert Expression().profiles == updated
            original = Path(config.EXPRESSION_PROFILE_PATH).read_bytes()
            def failed_install(src, dst):
                if src.endswith('.tmp'):
                    raise OSError(5, 'simulated write failure')
                fat_rename(src, dst)
            with patch.object(expression_module.os, 'rename', failed_install):
                try:
                    expression_module.save_profiles({'neutral': list(f)})
                except OSError as error:
                    assert error.args[0] == 5
                else:
                    raise AssertionError('Expected simulated failure')
            assert Path(config.EXPRESSION_PROFILE_PATH).read_bytes() == original
            real_rename(config.EXPRESSION_PROFILE_PATH, config.EXPRESSION_PROFILE_PATH + '.bak')
            assert Expression().profiles == updated  # Interrupted replacement recovery.
        print('PASS: FAT EEXIST replacement, repeated calibration, rollback and backup recovery')
        e.profiles = {label:[v+i*0.15 for v in f] for i,label in enumerate(LABELS)}
        e.reset()
        assert e.update(p,box,pose,1)['label'] == 'unknown'
        e.update(p,box,pose,1)
        assert e.update(p,box,pose,1)['label'] == 'neutral'
        e.profiles = {label:list(f) for label in LABELS}
        assert e.update(p,box,pose,1)['label'] == 'unknown'  # 相似样本拒识
        assert e.update(None,None,None,0)['label'] == 'unknown'
        for file in Path(d).iterdir():
            file.unlink()
        Path(d).rmdir()
    print('PASS: target association/loss/clipping; invariant features; quality gates; calibration persistence; debounce; ambiguity rejection')


if __name__ == '__main__':
    main()
