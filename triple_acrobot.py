"""Triple Acrobot task (3 segments)"""

import numpy as np
from numpy import cos, pi, sin
import gymnasium as gym
from gymnasium import Env, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

class TripleAcrobotEnv(Env):
    """
    ## Description
    Une version complexifiée de l'Acrobot classique, comportant 3 segments.
    Le système est sous-actionné : seul le couple (torque) entre le segment 1 et 2 est contrôlé.
    Le but est de balancer le 3ème segment au-dessus d'une hauteur cible.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 15,
    }

    dt = 0.05  # Pas de temps réduit pour + de précision

    # Paramètres physiques des 3 segments
    LINK_LENGTH_1 = 1.0  # [m]
    LINK_LENGTH_2 = 1.0  # [m]
    LINK_LENGTH_3 = 1.0  # [m]
    
    LINK_MASS_1 = 1.0    # [kg]
    LINK_MASS_2 = 1.0    # [kg]
    LINK_MASS_3 = 1.0    # [kg]
    
    LINK_COM_POS_1 = 0.5 # [m]
    LINK_COM_POS_2 = 0.5 # [m]
    LINK_COM_POS_3 = 0.5 # [m]
    
    LINK_MOI = 1.0       # Moment d'inertie

    MAX_VEL_1 = 4 * pi
    MAX_VEL_2 = 9 * pi
    MAX_VEL_3 = 15 * pi

    AVAIL_TORQUE = [-2.0, 0.0, +2.0] # augmenter pour compenser le poids supplémentaire du 3e segment

    torque_noise_max = 0.0
    SCREEN_DIM = 600

    def __init__(self, render_mode: str | None = None):
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.isopen = True
        
        # L'espace d'observation (mtn à 9 dimensions) : 
        # cos(t1), sin(t1), cos(t2), sin(t2), cos(t3), sin(t3), v1, v2, v3
        high = np.array( [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2, self.MAX_VEL_3], dtype=np.float32)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.state = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        low, high = utils.maybe_parse_reset_bounds(options, -0.1, 0.1)
        # Etat interne: [theta1, theta2, theta3, v1, v2, v3]
        self.state = self.np_random.uniform(low=low, high=high, size=(6,)).astype(np.float32)

        if self.render_mode == "human":
            self.render()
        return self._get_ob(), {}

    def step(self, a):
        s = self.state
        torque = self.AVAIL_TORQUE[a]

        if self.torque_noise_max > 0:
            torque += self.np_random.uniform(-self.torque_noise_max, self.torque_noise_max)

        # Augmentation de l'état pour passer à l'intégrateur (6 états + 1 action)
        s_augmented = np.append(s, torque)
        ns = rk4(self._dsdt, s_augmented, [0, self.dt])

        ns[0] = wrap(ns[0], -pi, pi)
        ns[1] = wrap(ns[1], -pi, pi)
        ns[2] = wrap(ns[2], -pi, pi)
        ns[3] = bound(ns[3], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[4] = bound(ns[4], -self.MAX_VEL_2, self.MAX_VEL_2)
        ns[5] = bound(ns[5], -self.MAX_VEL_3, self.MAX_VEL_3)
        
        self.state = ns
        terminated = self._terminal()
        reward = -1.0 if not terminated else 0.0

        if self.render_mode == "human":
            self.render()
            
        return self._get_ob(), reward, terminated, False, {}

    def _get_ob(self):
        s = self.state
        return np.array(
            [cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), cos(s[2]), sin(s[2]), s[3], s[4], s[5]], 
            dtype=np.float32
        )

    def _terminal(self):
        # calcul de la hauteur de l'extrémité du 3e segment
        # objectif est atteindre une hauteur > 2.0 (maximum théorique de 3.0)
        s = self.state
        hauteur = -cos(s[0]) - cos(s[1] + s[0]) - cos(s[2] + s[1] + s[0])
        return bool(hauteur > 2.0)

    def _dsdt(self, s_augmented):
        """
        Résolution matricielle des équations de mouvement pour un pendule triple.
        M(alpha) * dd_alpha + V(alpha, d_alpha) + G(alpha) = Tau
        """
        g = 9.81
        m1, m2, m3 = self.LINK_MASS_1, self.LINK_MASS_2, self.LINK_MASS_3
        l1, l2, l3 = self.LINK_LENGTH_1, self.LINK_LENGTH_2, self.LINK_LENGTH_3
        lc1, lc2, lc3 = self.LINK_COM_POS_1, self.LINK_COM_POS_2, self.LINK_COM_POS_3
        I1, I2, I3 = self.LINK_MOI, self.LINK_MOI, self.LINK_MOI
        
        a = s_augmented[-1]
        s = s_augmented[:-1]
        
        # Angles relatifs
        t1, t2, t3 = s[0], s[1], s[2]
        dt1, dt2, dt3 = s[3], s[4], s[5]
        
        # Angles absolus par rapport à la verticale
        a1 = t1
        a2 = t1 + t2
        a3 = t1 + t2 + t3
        
        da1 = dt1
        da2 = dt1 + dt2
        da3 = dt1 + dt2 + dt3
        
        # Construction de la Matrice de Masse (M)
        M11 = m1*lc1**2 + m2*l1**2 + m3*l1**2 + I1
        M22 = m2*lc2**2 + m3*l2**2 + I2
        M33 = m3*lc3**2 + I3
        
        c12 = (m2*l1*lc2 + m3*l1*l2)
        c13 = (m3*l1*lc3)
        c23 = (m3*l2*lc3)
        
        M12 = c12 * cos(a1 - a2)
        M13 = c13 * cos(a1 - a3)
        M23 = c23 * cos(a2 - a3)
        
        M = np.array([
            [M11, M12, M13],
            [M12, M22, M23],
            [M13, M23, M33]
        ])
        
        # Vecteur de forces centrifuges (V)
        V1 = c12 * sin(a1 - a2) * da2**2 + c13 * sin(a1 - a3) * da3**2
        V2 = -c12 * sin(a1 - a2) * da1**2 + c23 * sin(a2 - a3) * da3**2
        V3 = -c13 * sin(a1 - a3) * da1**2 - c23 * sin(a2 - a3) * da2**2
        V = np.array([V1, V2, V3])
        
        # Vecteur de Gravité (G)
        G1 = (m1*lc1 + m2*l1 + m3*l1) * g * sin(a1)
        G2 = (m2*lc2 + m3*l2) * g * sin(a2)
        G3 = (m3*lc3) * g * sin(a3)
        G = np.array([G1, G2, G3])
        
        # Couple appliqué : On actionne l'articulation 2 (entre segment 1 et 2)
        # La force agit négativement sur le lien 1, et positivement sur le lien 2
        Tau_abs = np.array([-a, a, 0.0])
        
        # Résolution du système linéaire: M * dd_a = Tau_abs - V - G
        RHS = Tau_abs - V - G
        dd_a = np.linalg.solve(M, RHS)
        
        # Reconversion des accélérations absolues en accélérations relatives
        ddt1 = dd_a[0]
        ddt2 = dd_a[1] - dd_a[0]
        ddt3 = dd_a[2] - dd_a[1]
        
        return dt1, dt2, dt3, ddt1, ddt2, ddt3, 0.0

    def render(self):
        if self.render_mode is None:
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled('pygame is not installed') from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.SCREEN_DIM, self.SCREEN_DIM))
            else:
                self.screen = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        surf.fill((255, 255, 255))
        s = self.state

        bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + self.LINK_LENGTH_3 + 0.2
        scale = self.SCREEN_DIM / (bound * 2)
        offset = self.SCREEN_DIM / 2

        if s is None:
            return None

        # Calcul des positions des 3 joints
        p1 = [
            -self.LINK_LENGTH_1 * cos(s[0]) * scale,
             self.LINK_LENGTH_1 * sin(s[0]) * scale,
        ]
        p2 = [
            p1[0] - self.LINK_LENGTH_2 * cos(s[0] + s[1]) * scale,
            p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1]) * scale,
        ]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [s[0] - pi / 2, s[0] + s[1] - pi / 2, s[0] + s[1] + s[2] - pi / 2]
        link_lengths = [self.LINK_LENGTH_1 * scale, self.LINK_LENGTH_2 * scale, self.LINK_LENGTH_3 * scale]

        # Ligne d'objectif
        pygame.draw.line(
            surf,
            start_pos=(-bound * scale + offset, 2.0 * scale + offset), # Cible = Hauteur 2.0
            end_pos=(bound * scale + offset, 2.0 * scale + offset),
            color=(0, 0, 0),
        )

        for (x, y), th, llen in zip(xys, thetas, link_lengths):
            x, y = x + offset, y + offset
            l, r, t, b = 0, llen, 0.1 * scale, -0.1 * scale
            coords = [(l, b), (l, t), (r, t), (r, b)]
            transformed_coords = []
            for coord in coords:
                coord = pygame.math.Vector2(coord).rotate_rad(th)
                coord = (coord[0] + x, coord[1] + y)
                transformed_coords.append(coord)
            
            # Dessin des segments
            gfxdraw.aapolygon(surf, transformed_coords, (0, 204, 204))
            gfxdraw.filled_polygon(surf, transformed_coords, (0, 204, 204))
            # Dessin des articulations
            gfxdraw.aacircle(surf, int(x), int(y), int(0.1 * scale), (204, 204, 0))
            gfxdraw.filled_circle(surf, int(x), int(y), int(0.1 * scale), (204, 204, 0))

        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False

def wrap(x, m, M):
    diff = M - m
    while x > M: x -= diff
    while x < m: x += diff
    return x

def bound(x, m, M=None):
    if M is None:
        M = m[1]
        m = m[0]
    return min(max(x, m), M)

def rk4(derivs, y0, t):
    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float64)
    else:
        yout = np.zeros((len(t), Ny), np.float64)

    yout[0] = y0

    for i in np.arange(len(t) - 1):
        this = t[i]
        dt = t[i + 1] - this
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0))
        k2 = np.asarray(derivs(y0 + dt2 * k1))
        k3 = np.asarray(derivs(y0 + dt2 * k2))
        k4 = np.asarray(derivs(y0 + dt * k3))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        
    # Retourne les 6 valeurs d'état (sans l'action ajoutée temporairement)
    return yout[-1][:6]