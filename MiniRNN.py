import torch
import torch.nn as nn
import torch.nn.functional as F


# MiniGRU 및 MiniLSTM의 공통적으로 사용하는 함수만 따로 묶어서 클래스화
class Utils:
    @staticmethod #일반적인 함수처럼 쓰려고 정의한 데코레이터
    # 이 함수는 수치적으로 안정적이지 않은 함수
    def parallel_scan(coeff, value):
        # coeff의 차원은 (bs, seq_len, input_size)
        # value의 차원은(bs, seq_len+1, input_size)
        
        #누적곱 연산 적용 후 pad 차원으로 (bs, seq_len+1, input_size)만들기
        cum_coeff = F.pad(torch.cumprod(coeff, dim=1), (0, 0, 1, 0)) 
        # 누적합 연산은 두 인자 차원이 (bs, seq_len+1, input_size)로
        # 같아졌으니 연산이 가능해진다.
        prefix = torch.cumsum(value * cum_coeff, dim=1)

        # 최종 연산 차원이(bs, seq_len+1, input_size)이니
        # 슬라이싱을 통해서 (bs, seq_len, input_size)만 리턴
        return prefix[:, 1:]

    @staticmethod
    def parallel_scan_log(log_coeff, log_value):
        # 입력인자 / 연산과정 / 출력과정의 차원은
        # parallel_scan와 모두 동일하며, 수치적안정성 개선 함수임
        a_star = F.pad(torch.cumsum(log_coeff, dim=1), (0, 0, 1, 0))
        log_h0_plus_b_star = torch.logcumsumexp(log_value - a_star, dim=1)
        log_prefix = a_star + log_h0_plus_b_star

        return torch.exp(log_prefix)[:, 1:]
    
    @staticmethod
    def g(x):   # h_tilde의 변환에 사용되는 함수
        return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))

    @staticmethod
    def log_g(x):   # h_tilde의 log공간 변환에 사용되는 함수
        return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))
    

# parallel mode만 구현하고 Sequence mode는 구현안함

class MiniGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MiniGRU, self).__init__()

        self.hidden_size = hidden_size

        # miniGRU는 게이트웨이 간소화->2개만 정의함
        self.cell_z = nn.Linear(input_size, hidden_size)
        self.cell_h = nn.Linear(input_size, hidden_size)

    def forward(self, x, h_0=None):
        bs, seq_len, _ = x.size() # (bs, seq_len, input_size)
        if h_0 is None: # 초기 hidden : (bs, 1, hidden_size)
            h_0 = torch.zeros(bs, 1, self.hidden_size,
                              device=x.device, dtype=x.dtype)
            
        k = self.cell_z(x) # z_t의 로그공간 연산으로 생성한 임시변수

        log_coeff = -F.softplus(k)

        log_z = -F.softplus(-k) #value의 첫번째 인자
        log_tilde_h = Utils.log_g(self.cell_h(x)) # value의 두번째 인자.
        log_h_0 = Utils.log_g(h_0) #첫 hidden인자 로그변환

        log_value = torch.cat([log_h_0, log_z*log_tilde_h], dim=1)

        #병렬 스캔 연산 수행 (수치 안정성 보장 함수로 사용)
        output = Utils.parallel_scan_log(log_coeff, log_value)

        return output # (bs, seq_len, input_size)
    


class MiniLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MiniLSTM, self).__init__()

        self.hidden_size = hidden_size

        # miniLSTM는 게이트웨이 간소화->3개만 정의함
        self.cell_f = nn.Linear(input_size, hidden_size)
        self.cell_i = nn.Linear(input_size, hidden_size)
        self.cell_h = nn.Linear(input_size, hidden_size)

    def forward(self, x, h_0=None):
        bs, seq_len, _ = x.size() # (bs, seq_len, input_size)
        if h_0 is None: # 초기 hidden : (bs, 1, hidden_size)
            h_0 = torch.zeros(bs, 1, self.hidden_size,
                              device=x.device, dtype=x.dtype)
        
        # f_prime, i_prime의 분모에 해당하는 인자의 로그변환 임시변수
        diff = F.softplus(-self.cell_f(x)) - F.softplus(-self.cell_i(x))

        log_f_prime = -F.softplus(diff) #이게 log_coeff
        log_i_prime = -F.softplus(-diff) #log_value의 첫번째 인자

        log_tilde_h = Utils.log_g(self.cell_h(x)) # value의 두번째 인자.
        log_h_0 = Utils.log_g(h_0) #첫 hidden인자 로그변환

        log_value = torch.cat([log_h_0, log_i_prime * log_tilde_h], dim=1)

        #병렬 스캔 연산 수행 (수치 안정성 보장 함수로 사용)
        output = Utils.parallel_scan_log(log_f_prime, log_value)

        return output # (bs, seq_len, input_size)
