# 在 IDE 修改 LABEL 后在线运行；写入一次标定请求，然后重新运行 main.py。
# 主程序启动后保持正脸，3 秒准备、15 次有效采样，终端提示 saved 后完成。
import ujson
LABEL = 'neutral'  # neutral / happy / angry / sad / surprise
with open('/sdcard/app/MyVisionHub/expression_request.json', 'w') as f:
    f.write(ujson.dumps({'label': LABEL}))
print('Calibration queued:', LABEL, '- now run MyVisionHub main.py')
