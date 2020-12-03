import numpy as np
import math as mathf

class Measurement:
    def __init__(self, s):
        self.s = s
        self.mag = np.array([0, 0, 0], dtype=float)
        self.linearAccel = np.array([[0, 0, 0]], dtype=float)
        self.rotRate = np.array([[0, 0, 0, 0]], dtype=float)
        self.wPrev = np.array([0, 0, 0]) # Vector containing previous time step rotation rate
        self.qInit = np.array([1,0,0,0])
        self.timeValue1 = 0
        self.initialized = False
        self.do_bias_estimation = True
        self.biasAlpha = 0.01
        self.wx_bias = 0
        self.wy_bias = 0
        self.wz_bias = 0
        self.Gravity = 9.81
        self.wThreshhold = 0.2
        self.dwThreshhold = 0.01
        self.accThreshold = 0.1
        self.gamma = 0.01
        self.epsilon = 0.9


    def quatProd(self,a, b):
        # Input: two quaternions
        # Output: Quaternion product
        b.shape = (4,)
        a.shape = (4,)
        ab = np.zeros(4, )
        ab[0] = np.dot(a[0], b[0]) - np.dot(a[1], b[1]) - np.dot(a[2], b[2]) - np.dot(a[3], b[3])
        ab[1] = np.dot(a[0], b[1]) + np.dot(a[1], b[0]) + np.dot(a[2], b[3]) - np.dot(a[3], b[2])
        ab[2] = np.dot(a[0], b[2]) - np.dot(a[1], b[3]) + np.dot(a[2], b[0]) + np.dot(a[3], b[1])
        ab[3] = np.dot(a[0], b[3]) + np.dot(a[1], b[2]) - np.dot(a[2], b[1]) + np.dot(a[3], b[0])
        return ab

    def quat_to_ypr(self,q):
        # Converts state quaternion to euler angles use a yaw, pitch, roll sequence.

        yaw = mathf.atan2(2.0 * (q[1] * q[2] + q[0] * q[3]), q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3])
        pitch = -mathf.sin(2.0 * (q[1] * q[3] - q[0] * q[2]))
        roll = mathf.atan2(2.0 * (q[0] * q[1] + q[2] * q[3]), q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3])

        yaw -= -0.13

        return np.array([roll, pitch, yaw])

    #
    def quaternConj(self,q):
        # Returns quaternion conjugate
        q.shape = (4,)
        qConj = np.array([q[0], -q[1], -q[2], -q[3]])
        return qConj

    def collectIMU(self,message):
        # Inputs: String containing all of the sensor data in utf-8 format
        # Output: Numpy array containing the sensor data for each magnetometer, accelerometer, gyrcscope
        t1 = message.decode("utf-8").replace(',', '').split()[0]  # Extract timestamp
        list = message.decode("utf-8").replace(' ', '').split(',')
        try:
            if list.index('3'):
                accelFlag = 1
                accelInd = list.index('3')
                linearAccelValues = np.array(
                    [float(list[accelInd + 1]), float(list[accelInd + 2]), float(list[accelInd + 3])])
                linearAccelValues.shape = (1, 3)
        except(ValueError):
            # linearAccelValues = self.linearAccel[len(self.linearAccel) - 1]
            linearAccelValues = self.linearAccel
            linearAccelValues.shape = (1, 3)

        try:
            if list.index('5'):
                magFlag = 1
                magInd = list.index('5')
                magValues = np.array([float(list[magInd + 1]), float(list[magInd + 2]), float(list[magInd + 3])])
                magValues.shape = (1, 3)
        except(ValueError):
            # magValues = self.mag[len(self.mag) - 1]
            magValues = self.mag
            magValues.shape = (1, 3)

        try:
            if list.index('4'):
                rotFlag = 1
                rotInd = list.index('4')
                rotRateValues = np.array(
                    [float(list[rotInd + 1]), float(list[rotInd + 2]), float(list[rotInd + 3])])
                rotRateValues = np.array([0, rotRateValues[0], rotRateValues[1], rotRateValues[2]])
                rotRateValues.shape = (1, 4)
        except(ValueError):
            # rotRateValues = self.rotRate[len(self.rotRate) - 1]
            rotRateValues = self.rotRate
            rotRateValues.shape = (1, 4)

        return t1, rotRateValues, magValues, linearAccelValues

    def quatRotate(self,x, y, z, q0, q1, q2, q3):
        '''
        Rotates a 3D vector by a quaternion
        Input: x, y, z vector components and state quaternion
              x = x vector component
              y = y vector component
              z = z vector component
              q0 = scalar quaternion value
              q1 = first quaternion vector component
              q2 = second quaternion vector component
              q3 = third quaternion vector component

        Output: Rotated vector
        '''

        vx = (q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3) * x + 2 * (q1 * q2 - q0 * q3) * y + 2 * (q1 * q3 + q0 * q2) * z
        vy = 2 * (q1 * q2 + q0 * q3) * x + (q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3) * y + 2 * (q2 * q3 - q0 * q1) * z
        vz = 2 * (q1 * q3 - q0 * q2) * x + 2 * (q2 * q3 + q0 * q1) * y + (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3) * z
        return vx, vy, vz

    def predictQuat(self,qIn, wx, wy, wz, wx_bias, wy_bias, wz_bias, dt):
        # Predicts state quaternion from rotation rates.
        q0 = qIn[0]
        q1 = qIn[1]
        q2 = qIn[2]
        q3 = qIn[3]

        wx_unb = wx - wx_bias
        wy_unb = wy - wy_bias
        wz_unb = wz - wz_bias

        q0_pred = q0 + 0.5 * dt * (wx_unb * q1 + wy_unb * q2 + wz_unb * q3)
        q1_pred = q1 + 0.5 * dt * (-wx_unb * q0 - wy_unb * q3 + wz_unb * q2)
        q2_pred = q2 + 0.5 * dt * (wx_unb * q3 - wy_unb * q0 - wz_unb * q1)
        q3_pred = q3 + 0.5 * dt * (-wx_unb * q2 + wy_unb * q1 - wz_unb * q0)

        qOut = np.array([q0_pred, q1_pred, q2_pred, q3_pred])
        qOut = qOut / np.linalg.norm(qOut)
        return qOut

    def scaleQuat(self,epsilon, gain, quat):
        # scale and smooth quaternion
        angle = np.arccos(quat[0])
        if quat[0] < 0:  # EPSILON?
            angle = np.arccos(quat[0])
            A = np.sin(angle * (1 - gain)) / np.sin(angle)
            B = np.sin(angle * gain) / np.sin(angle)
            dq0 = A + B * quat[0]
            dq1 = B * quat[1]
            dq2 = B * quat[2]
            dq3 = B * quat[3]

        else:  # Linear Interpolation
            dq0 = (1 - gain) + gain * quat[0]
            dq1 = gain * quat[1]
            dq2 = gain * quat[2]
            dq3 = gain * quat[3]

        quatOut = np.array([dq0, dq1, dq2, dq3])
        quatOut = quatOut / np.linalg.norm(quatOut)

        return quatOut

    def correctAcc(self,a, quat):
        # Corrects to quaternion estimate using accelerometer data by calculating gravity vector (true down)
        a = a / np.linalg.norm(a)

        # Acc reading rotated into the world frame by the inverse predicted quaternion (predicted gravity)
        gx, gy, gz = self.quatRotate(a[0], a[1], a[2], quat[0], -quat[1], -quat[2],
                                -quat[3])

        dq0 = np.sqrt((gz + 1) * .5)
        dq1 = -gy / (2 * dq0)
        dq2 = gx / (2 * dq0)
        dq3 = 0

        dquat = np.array([dq0, dq1, dq2, dq3])

        return dquat

    def correctMag(self,m, q):
        # Estimates magnetic field quaternion to correct the state quaternion estimate for the heading angle
        lx, ly, lz = self.quatRotate(m[0], m[1], m[2], q[0], -q[1], -q[2], -q[3])

        # Delta quaternion that rotates the l so that it lies in the xz-plane (points north)
        gamma = lx * lx + ly * ly
        beta = np.sqrt(gamma + lx * np.sqrt(gamma))

        dq0 = beta / (np.sqrt(2 * gamma))
        dq1 = 0
        dq2 = 0
        dq3 = ly / (np.sqrt(2) * beta)

        dq = np.array([dq0, dq1, dq2, dq3])
        return dq

    def updateBiases(self,a, w, wPrev, wx_bias, wy_bias, wz_bias, bias_alpha):
        # Update gyroscope sensor biases when called
        steady_state = self.checkState(a, w, wPrev)
        wx = w[0]
        wy = w[1]
        wz = w[2]

        if steady_state:
            wx_bias = wx_bias + bias_alpha * (wx - wx_bias)
            wy_bias = wy_bias + bias_alpha * (wy - wy_bias)
            wz_bias = wz_bias + bias_alpha * (wz - wz_bias)

        wx_prev = wx
        wy_prev = wy
        wz_prev = wz

        wPrev = np.array([wx_prev, wy_prev, wz_prev])
        wBias = np.array([wx_bias, wy_bias, wz_bias])

        return wPrev, wBias

    def checkState(self,a, w, wPrev):
        # Checks if sensors are at steady state before estimating bias
        ax = a[0]
        ay = a[1]
        az = a[2]
        wx = w[0]
        wy = w[1]
        wz = w[2]

        wx_prev = wPrev[0]
        wy_prev = wPrev[1]
        wz_prev = wPrev[2]
        acc_magnitude = np.sqrt(ax * ax + ay * ay + az * az)

        if (abs(acc_magnitude - self.Gravity) > self.accThreshold):
            return False

        if (abs(wx - wx_prev) > self.dwThreshhold or
                abs(wy - wy_prev) > self.dwThreshhold or
                abs(wz - wz_prev) > self.dwThreshhold):
            return False

        if (abs(wx - self.wx_bias) > self.wThreshhold or
                abs(wy - self.wy_bias) > self.wThreshhold or
                abs(wz - self.wz_bias) > self.wThreshhold):
            return False

        return True

    def ep2Rot(self,q):
        # Converts quaternion to rotation matrix
        X = q[1]
        Y = q[2]
        Z = q[3]
        W = q[0]

        xx = X * X
        xy = X * Y
        xz = X * Z
        xw = X * W

        yy = Y * Y
        yz = Y * Z
        yw = Y * W

        zz = Z * Z
        zw = Z * W

        m00 = 1 - 2 * (yy + zz)
        m01 = 2 * (xy - zw)
        m02 = 2 * (xz + yw)

        m10 = 2 * (xy + zw)
        m11 = 1 - 2 * (xx + zz)
        m12 = 2 * (yz - xw)

        m20 = 2 * (xz - yw)
        m21 = 2 * (yz + xw)
        m22 = 1 - 2 * (xx + yy)

        '''
        m00 = 1 - 2 * (yy + zz)
        m01 = 2 * (xy - zw)
        m02 = 2 * (xz + yw)

        m10 = 2 * (xy + zw)
        m11 = 1 - 2 * (xx + zz)
        m12 = 2 * (yz - xw)

        m20 = 2 * (xz - yw)
        m21 = 2 * (yz + xw)
        m22 = 1 - 2 * (xx + yy)
        '''

        rotMat = np.array([[m00, m01, m02, 0], [m10, m11, m12, 0], [m20, m21, m22, 0], [0, 0, 0, 1]])
        return rotMat.T
    def eulerAngles(self):
        eulerOut = self.quat_to_ypr(self.quaternConj(self.qInit))
        return eulerOut

    def updatePose(self):
        message, address = self.s.recvfrom(8192)  # Get message
        [t1, rotRateValues, magValues, linearAccelValues] = self.collectIMU(message)  # Convert string to arrays

        # Clear acceleration array to preserve memory
        # if len(self.linearAccel) > 500:
        #     self.linearAccel = self.linearAccel[(len(self.linearAccel) - 2):]
        #     self.mag = self.mag[(len(self.mag) - 2):]
        #     self.rotRate = self.rotRate[(len(self.rotRate) - 2):]
        #
        # self.linearAccel = np.concatenate((self.linearAccel, linearAccelValues), axis=0)
        # self.mag = np.concatenate((self.mag, magValues), axis=0)
        # self.rotRate = np.concatenate((self.rotRate, rotRateValues), axis=0)

        timeValue2 = float(t1)
        dt = timeValue2 - self.timeValue1
        self.timeValue1 = timeValue2

        currAccel = linearAccelValues[0]

        currMag = magValues[0]
        currMag = np.array([currMag[0] / 10 ** 6, currMag[1] / 10 ** 6, currMag[2] / 10 ** 6])

        if not self.initialized:
            if np.sum(currMag) == 0:
                return
            tempAccel = currAccel / np.linalg.norm(currAccel)
            mx = currMag[0]
            my = currMag[1]
            mz = currMag[2]
            if tempAccel[2] >= 0:
                qtemp0 = np.sqrt((tempAccel[2] + 1) * 0.5)
                qtemp1 = -tempAccel[1] / (2 * qtemp0)
                qtemp2 = tempAccel[0] / (2 * qtemp0)
                qtemp3 = 0
            else:
                X = np.sqrt((1 - tempAccel[2]) * 0.5)
                qtemp0 = -tempAccel[1] / (2 * X)
                qtemp1 = X
                qtemp2 = 0
                qtemp3 = tempAccel[0] / (2 * X)

            qtempAccel = np.array([qtemp0, qtemp1, qtemp2, qtemp3])
            lx = (qtemp0 * qtemp0 + qtemp1 * qtemp1 - qtemp2 * qtemp2) * mx + 2.0 * (qtemp1 * qtemp2) * my - 2.0 * (
                    qtemp0 * qtemp2) * mz
            ly = 2.0 * (qtemp1 * qtemp2) * mx + (qtemp0 * qtemp0 - qtemp1 * qtemp1 +
                                                 qtemp2 * qtemp2) * my + 2.0 * (qtemp0 * qtemp1) * mz
            gamma = lx * lx + ly * ly
            beta = np.sqrt(gamma + lx * np.sqrt(gamma))
            q0_mag = beta / (np.sqrt(2.0 * gamma))
            q3_mag = ly / (np.sqrt(2.0) * beta)
            qTempMag = np.array([q0_mag, 0, 0, q3_mag])

            self.qInit = self.quatProd(qtempAccel, qTempMag)

            self.initialized = True
            return

        # Correction from ACC

        # Prediction
        w = rotRateValues[0]
        wShort = np.array([w[1], w[2], w[3]])
        if self.do_bias_estimation:
            self.wPrev, wBias = self.updateBiases(currAccel, wShort, self.wPrev, self.wx_bias, self.wy_bias, self.wz_bias, self.biasAlpha)
            self.wx_bias = wBias[0]
            self.wy_bias = wBias[1]
            self.wz_bias = wBias[2]

        # Returns normalized quaternion prediction from angular velocity vector and prev. quaternion
        quatPredict = self.predictQuat(self.qInit.reshape(4, ), wShort[0], wShort[1], wShort[2], self.wx_bias, self.wy_bias, self.wz_bias,
                                  dt)

        deltaACC = self.correctAcc(currAccel, quatPredict)

        # alpha = getAdaptiveGain(gain_acc_, ax, ay, az)
        alpha = 0.01

        # Returns normalized quaternion
        deltaqScale = self.scaleQuat(self.epsilon, alpha, deltaACC)

        self.qInit = self.quatProd(quatPredict, deltaqScale)

        # Correction from mag
        if np.sum(currMag) != 0:
            deltaQMag = self.correctMag(currMag, self.qInit)

            deltaQMagScale = self.scaleQuat(self.epsilon, alpha, deltaQMag)
            self.qInit = self.quatProd(self.qInit, deltaQMagScale)

        self.qInit = self.qInit / np.linalg.norm(self.qInit)
