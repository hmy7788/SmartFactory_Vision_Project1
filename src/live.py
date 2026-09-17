"""브라우저 카메라로 실시간 판정 — streamlit-webrtc가 깔려 있을 때만 쓴다.

무엇이 다른가: 지금까지의 웹캠은 **서버(streamlit을 켠 컴퓨터)**의 카메라를 열었다. 그래서 팀원이
내 주소로 접속하면 팀원 화면에 내 카메라가 보였다. webrtc는 **보는 사람의 브라우저** 카메라를 써서
영상을 서버로 보낸다 — 각자 자기 카메라로 판정받는다.

카메라 권한의 전제: 브라우저는 주소가 https이거나 localhost일 때만 카메라를 내준다(보안 컨텍스트).
같은 와이파이의 http://192.168.x.x:8501로 들어오면 권한 창이 아예 안 뜬다. 코드로 우회할 수 없고,
터널로 https 주소를 만들어야 한다. 발표 전에 반드시 실제 기기로 한 번 확인할 것.

왜 판정을 따로 떼어놨나 — webrtc는 프레임을 별도 스레드로 준다. 거기서는 st.*를 부를 수 없고
(Streamlit은 스크립트 스레드에만 붙는다), 0.3초짜리 추론을 거기서 돌리면 그만큼 영상이 멈춘다.
그래서 이 파일이 하는 일은 둘뿐이다:
  - 받은 프레임은 손대지 않고 그대로 돌려준다  → 영상은 카메라 속도대로 부드럽게 흐른다
  - 판정은 작업 스레드가 '가장 최근 프레임' 하나만 집어서 돌리고 결과를 잠금 뒤에 놓는다
영상 프레임 수와 판정 횟수를 분리하는 것이 핵심이다. CPU에서 초당 3번 판정해도 영상은 30fps다.
쌓인 프레임을 버리는 이유도 같다 — 다음 판정은 더 새로운 장면으로 하는 것이 맞다.
"""
import threading

try:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
    AVAILABLE = True
except Exception:          # 패키지 없음, 또는 av/aiortc 설치 실패
    AVAILABLE = False

# 공용 STUN. 같은 기기(localhost)면 없어도 붙지만, 다른 기기에서 들어오면 서로의 주소를 찾는 데 필요하다
RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


class LiveJudge:
    """webrtc 프레임을 받아 영상은 그대로 통과시키고, 뒤에서 최신 프레임만 판정한다."""

    def __init__(self, predict):
        self._predict = predict           # (image_bgr) -> {"pred":…, "infer_ms":…, "error":…}
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._pending = None              # 아직 판정 안 한 최신 프레임 (덮어쓴다)
        self._result = None
        self._frames = 0                  # 카메라가 보낸 장수
        self._judged = 0                  # 실제로 판정한 장수
        self._stop = False
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    # ---- webrtc 프레임 스레드가 부른다. 무거운 일을 하면 안 되는 자리 ----
    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        with self._lock:
            self._pending = image         # 앞의 것을 덮는다 = 밀린 프레임은 판정하지 않는다
            self._frames += 1
            self._wake.notify()
        return frame                      # 손대지 않고 그대로 돌려준다

    # ---- 판정 작업 스레드 ----
    def _loop(self):
        while True:
            with self._lock:
                while self._pending is None and not self._stop:
                    self._wake.wait(0.5)
                if self._stop:
                    return
                image, self._pending = self._pending, None
            # 잠금 밖에서. 여기가 0.3초짜리다.
            # 예외를 삼키는 이유: 이 스레드가 죽으면 화면은 마지막 판정을 그대로 물고 멈춘다 —
            # 틀린 결과를 계속 보여주면서 아무 표시도 없는 것이 제일 나쁘다. 화면에 실패로 띄운다.
            try:
                out = self._predict(image)
            except Exception as e:
                out = {"pred": None, "infer_ms": None, "error": f"{type(e).__name__}: {e}"}
            with self._lock:
                self._result = out
                self._judged += 1

    # ---- 화면(스크립트 스레드)이 읽는다 ----
    def snapshot(self) -> tuple[dict | None, int, int]:
        """(마지막 판정 결과, 받은 프레임 수, 판정한 수). 아직 한 번도 안 끝났으면 결과는 None."""
        with self._lock:
            return self._result, self._frames, self._judged

    def stop(self) -> None:
        with self._lock:
            self._stop = True
            self._wake.notify_all()

    # streamlit-webrtc가 정리할 때 부른다
    on_ended = stop


def stream(predict, key: str = "live", width: int = 640):
    """webrtc 스트림을 띄우고 context를 돌려준다. ctx.video_processor가 LiveJudge."""
    return webrtc_streamer(
        key=key,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": {"width": {"ideal": width}}, "audio": False},
        video_processor_factory=lambda: LiveJudge(predict),
        async_processing=True,
    )
