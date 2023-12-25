"""
시뮬링크-카메이커와 직접 연동을 위한 기능이 담긴 예제 코드
"""

import matlab.engine
import socket
import struct
import time
import threading
import getpass

from queue import Queue

# 카메이커(시뮬링크)와 통신을 위한 TCP/IP 쓰레드소소막창 본점
def tcp_thread(ip, port, send_queue, receive_queue, action_num=3, state_num=1):
    BUFFER_SIZE = state_num * 8 # 수신할 데이터의 갯수 * 8

    # 시뮬링크와 통신할 TCP/IP
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((ip, port))

    s.listen(1)

    print("Waiting for connection...", port)
    conn, addr = s.accept()
    print("Connection from:", addr)

    # 카메이커의 preparation 단계 이후 첫 데이터를 수신하게끔 하기 위해 sleep
    time.sleep(1.5)
    
    while True:
        try:            
            # 전송할 데이터 수집
            data_to_send = send_queue.get()
            
            # 데이터 전송
            struct_format = '!{}d'.format(action_num) # !1d에서 1은 수신할 데이터 갯수
            data = struct.pack(struct_format, *data_to_send)
            conn.send(data)

            # 데이터 수신
            received_data = conn.recv(BUFFER_SIZE)
            if not received_data:
                break
            
            # 수신한 데이터 메인 쓰레드로 전송
            try:
                struct_format = '!{}d'.format(state_num) # !1d에서 1은 수신할 데이터 갯수
                unpacked_data = struct.unpack(struct_format, received_data)
                receive_queue.put(unpacked_data)
            except:
                receive_queue.put(False)
            
        except Exception as e:
            # 연결 비정상 종료
            print("Connection killed.", e)
            break
    
    # 카메이커와의 통신 종료
    print("Connection closed.", port)
    receive_queue.put(False)

    conn.close()    

# 카메이커 시뮬링크와 직접 통신하기 위한 클라스
class CMcontrolNode:
    def __init__(self, action_queue, state_queue, action_num, state_num,
                 host='127.0.0.1', port=10003, matlab_path='C:/CM_Projects/JX1_102/src_cm4sl', simul_path='pythonCtrl_DLC2'):
        # 카메이커 프로젝트 폴더 밑 src_cm4sl 폴더
        self.MATLAB_PATH = matlab_path
        self.SIMUL_PATH = simul_path

        # TCP/IP. 예제 코드에서는 localhost에서 실행하도록 함
        self.host = host # 127.0.0.1
        self.port = port # 포트 번호는 process rank에 따라 자동 적용. 10000부터 1씩 늘어남

        # RL 에이전트로부터 수신/송신할 action/state 전송용 queue
        self.action_queue = action_queue
        self.state_queue = state_queue

        # TCP/IP 통신 시 pack/unpack할 데이터의 수를 알기 위해 적용함
        # Gym Env에서 자동 계산되어 적용
        self.action_num = action_num
        self.state_num = state_num

        # 매틀랩, 시뮬링크, 카메이커 실행
        self._setup_sim()

    def __del__(self):
        self.kill()

    def kill(self):
        # 프로세스 종료 후 처리
        self.eng.eval("cmguicmd('GUI quit')", nargout=1)
        print("Carmaker GUI closed.")
        # self.eng.close_system('pythonCtrl', nargout=0)
        # print("close system")
        self.eng.quit()
        print("bye")

    def _setup_sim(self):
        # 매틀랩 시작 및 경로 지정
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.MATLAB_PATH)
        self.eng.cd(self.MATLAB_PATH)
        print("connected to MATLAB")

        self.eng.Load_Parameters_RWS_simple_v1(nargout=0)

        # 시뮬링크 실행
        self.model = self.eng.load_system(self.SIMUL_PATH)
        print("opened pythonCtrl.slx")

        # 카메이커 GUI 실행
        carmaker_gui_path = self.SIMUL_PATH+'/Open CarMaker GUI'
        self.eng.open_system(carmaker_gui_path,nargout=0)
        print("opened CarMaker GUI")

        # TestRun 실행. 같은 프로젝트 내의 다른 테스트런 실행 가능. 예제 : Python_Test
        self.eng.eval("cmguicmd('LoadTestRun \"Python_Test\"')", nargout=0)
        print("Loaded testrun")

        # 통신을 위한 포트 지정
        print("set_param('{}/CarMaker/VehicleControl/CreateBus VhclCtrl/tcpiprcv', 'Port', '".format(self.SIMUL_PATH)+str(self.port)+"')")
        self.eng.set_param('{}/CarMaker/VehicleControl/CreateBus VhclCtrl/tcpiprcv'.format(self.SIMUL_PATH), 'Port', str(self.port), nargout=0)
        print("set_param({}'/CarMaker/CM_LAST/tcpipsend', 'Port', '".format(self.SIMUL_PATH) + str(self.port)+"')")
        self.eng.set_param('{}/CarMaker/CM_LAST/tcpipsend'.format(self.SIMUL_PATH), 'Port', str(self.port), nargout=0)

        time.sleep(1)
        print("Click Enter To Start")

    def start_sim(self):
        # TCP/IP 쓰레드 데이터 전송용 큐 생성
        send_queue = Queue()

        # TCP/IP 쓰레드 데이터 수신용 큐 생성
        receive_queue = Queue()

        # TCP/IP 스레드 생성 및 실행
        t = threading.Thread(target=tcp_thread, daemon=True, args=(self.host, self.port, send_queue, receive_queue, self.action_num, self.state_num))
        t.start()

        # 카메이커 시뮬레이션 Start 명령어 전송
        self.eng.set_param(self.model, 'SimulationCommand', 'start', nargout=0)
        print("Simulation started")

        while True:
            # 데이터 생성 및 전송
            data_to_send = self.action_queue.get()
            # print("ACTION rcvd", data_to_send)
            send_queue.put(data_to_send)
            # print("Data sent")

            # 수신한 데이터 처리
            received_data = receive_queue.get()
            # print("Main rcvd", received_data)
            self.state_queue.put(received_data)
            # 수신한 데이터를 처리하는 코드 작성

            # 시뮬레이션이 실행되고 있는지 여부 확인
            if t.is_alive() == True and received_data != False:
                pass
            else:
                break

        print("Simulation Stopped.")
        # t.stop()

    def stop_sim(self):
        # 시뮬레이션 Stop 명령어 전송
        self.eng.set_param(self.model, 'SimulationCommand', 'stop', nargout=0)
        time.sleep(1)

if __name__ == "__main__":
    # 카메이커 연결 노드 클라스가 정상 작동하는지 알아보는 테스트 코드
    def cm_thread(port, action_queue, state_queue):
        cm_env = CMcontrolNode(port=port, action_queue=action_queue, state_queue=state_queue, action_num=2, state_num=4)
        cm_env.start_sim()
        cm_env.stop_sim()

    action_queue = Queue()
    state_queue = Queue()

    t1 = threading.Thread(target=cm_thread, daemon=True, args=(10001,action_queue,state_queue,))
    # t2 = threading.Thread(target=cm_thread, args=(10002,))

    t1.start()
    # t2.start()

    while True:
        action_queue.put([1.0, 0.0])
        state = state_queue.get()
        print("state: ", state)
        if state == False:
            print("End")
            break

    print("Waiting for thread..")
    t1.join()
    # t2.join()





