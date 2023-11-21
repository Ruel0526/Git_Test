import numpy as np
import random
import math
import scipy.special as sp


class DRLenvMIMO(object):
    def __init__(self):
        self.antenna = 10
        self.users =  4
        self.transmitters = 19

        self.tx_position = self.tx_positions_gen()
        self.rx_position = self.rx_positions_gen(self.tx_position)



    def tx_positions_gen(self):
        R = 500
        tx_positions = []

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
        '''
        return tx_positions

    def rx_positions_gen(self, tx_position):
        rx_positions=[]


        #rx_positions = [[None for j in range(cols)] for i in range(rows)]

        r = 200
        R = 500
        ran = R * np.sqrt(3) / 2
        for i in range(self.transmitters):
            x1, y1 = tx_position[i]


            rx_positions_temp = []
            for j in range(self.users):
                x2 = x1
                y2 = y1

                while np.sqrt((x2-x1) ** 2 + (y2-y1) ** 2) < r:
                    x2 = random.uniform(x1 - ran, x1 + ran)
                    y2 = random.uniform(y1 - ran, y1 + ran)

                rx_positions_temp.append((x2, y2))
            rx_positions.append(rx_positions_temp)


        return rx_positions

    def Jakes_channel(self, previous_channel): # 유저 한명의 채널

        f_d = 10
        T = 0.2
        rho = sp.jv(0, 2*math.pi*f_d*T)

        channel_vector = np.zeros((1,self.antenna), dtype = complex)


        for k in range(self.antenna):



            innov = random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j

            #if previous_channel == initial:
            #    h = random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j



            h = rho * previous_channel[0,k] + (math.sqrt(1-math.pow(rho, 2)) * innov)

            #channel = math.pow(np.absolute(h), 2) * pathloss

            channel_vector[0,k] = h

        return channel_vector


    def channel_gain(self,  tx_position, rx_position, small_scale):

        x1, y1 = tx_position
        x2, y2 = rx_position

        d_k = np.sqrt((x2-x1) ** 2 + (y2-y1) ** 2)
        rho_0 = 61.39  # dB scale. 참조거리 1m, frequency 28GHz 기준. https://www.immersionrc.com/rf-calculators/ 참조
        alpha = 2  # pathloss exponent; free space is assumed
        pathloss = rho_0 * (d_k) ** (-alpha)

        gain = np.zeros((1,self.antenna))
        for i in range(self.antenna):

            gain[0,i] = math.pow(np.absolute(small_scale[i]), 2) * pathloss

        return gain

