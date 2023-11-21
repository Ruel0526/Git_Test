import numpy as np
import random
import math
import scipy.special as sp


class DRLenv(object):
    def __init__(self):


        self.transmitters = 3
        self.tx_position = self.tx_positions_gen()
        self.rx_position = self.rx_positions_gen(self.tx_position)

    def tx_positions_gen(self):
        R = 500
        tx_positions = []
        '''
        tx_positions.append((-2*R, 2*R * math.sqrt(3)))
        tx_positions.append((0, 2*R * math.sqrt(3)))
        tx_positions.append((2*R, 2*R * math.sqrt(3)))

        tx_positions.append((-3*R, R * math.sqrt(3)))
        tx_positions.append((-R, R * math.sqrt(3)))
        tx_positions.append((R, R * math.sqrt(3)))
        tx_positions.append((3*R, R * math.sqrt(3)))

        tx_positions.append((-4*R, 0))
        tx_positions.append((-2 * R, 0))
        tx_positions.append((0, 0))
        tx_positions.append((2 * R, 0))
        tx_positions.append((4 * R, 0))

        tx_positions.append((-3 * R, -R * math.sqrt(3)))
        tx_positions.append((-R, -R * math.sqrt(3)))
        tx_positions.append((R, -R * math.sqrt(3)))
        tx_positions.append((3 * R, -R * math.sqrt(3)))

        tx_positions.append((-2 * R, -2 * R * math.sqrt(3)))
        tx_positions.append((0, -2 * R * math.sqrt(3)))
        tx_positions.append((2 * R, -2 * R * math.sqrt(3)))
        '''
        tx_positions.append((-2 * R, 0))
        tx_positions.append((0, 0))
        tx_positions.append((-R, -R * math.sqrt(3)))
        
        return tx_positions

    def rx_positions_gen(self, tx_position):
        rx_positions=[]
        r = 200
        R = 500
        ran = R * np.sqrt(3) / 2
        for i in range(self.transmitters):
            x1, y1 = tx_position[i]
            x2 = x1
            y2 = y1

            while np.sqrt((x2-x1) ** 2 + (y2-y1) ** 2) < r:
                x2 = random.uniform(x1 - ran, x1 + ran)
                y2 = random.uniform(y1 - ran, y1 + ran)

            rx_positions.append((x2, y2))

        return rx_positions

    def Jakes_channel(self, previous_channel): # 유저 한명의 채널

        f_d = 10
        T = 0.02
        rho = sp.jv(0, 2*math.pi*f_d*T)


        initial = np.zeros(10)



        innov = random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j

            #if previous_channel == initial:
            #    h = random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j



        h = rho * previous_channel + (math.sqrt(1-math.pow(rho, 2)) * innov)

            #channel = math.pow(np.absolute(h), 2) * pathloss

        channel_vector = h

        return channel_vector


    def channel_gain(self,  tx_position, rx_position, small_scale):

        x1, y1 = tx_position
        x2, y2 = rx_position

        d_k = np.sqrt((x2-x1) ** 2 + (y2-y1) ** 2)
        PL_0 = 120.9

        log_normal = 8
        pathloss = PL_0 + 37.6 * math.log10(d_k/1000) + random.gauss(0, log_normal)

        gain = small_scale.conjugate() * small_scale / (10 ** (pathloss / 10))# / (10 ** (log_normal / 10))

        return np.real(gain)

