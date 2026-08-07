# gui/renderer.py

import math
from typing import Optional

import numpy as np
import pygame


class MazeRenderer:
    def __init__(self, env, cell_size=28, side_panel=320, margin=12):
        pygame.init()
        pygame.font.init()
        self.env = env
        self.cell_size = cell_size
        self.side_panel = side_panel
        self.margin = margin

        self.rows = env.rows
        self.cols = env.cols

        self.width = margin * 2 + self.cols * cell_size + side_panel
        self.height = margin * 2 + self.rows * cell_size

        self.font_sm = pygame.font.SysFont("consolas", 14)
        self.font_md = pygame.font.SysFont("consolas", 18)
        self.font_lg = pygame.font.SysFont("consolas", 22)

        # Colors
        self.COLORS = {
            "bg": (22, 24, 29),
            "grid": (40, 44, 52),
            "wall": (60, 63, 65),
            "start": (0, 160, 255),
            "goal": (30, 200, 70),
            "key": (255, 215, 0),
            "penalty": (220, 60, 60),
            "energy": (130, 200, 255),
            "agent": (255, 255, 255),
            "panel": (29, 31, 38),
            "text": (220, 220, 220),
            "accent": (100, 180, 255),
        }

        # Overlays
        self.value_grid: Optional[np.ndarray] = None
        self.policy_grid: Optional[np.ndarray] = None
        self.overlay_alpha = 140  # 0..255

    def set_env(self, env):
        self.env = env
        self.rows = env.rows
        self.cols = env.cols

    def set_overlays(self, value_grid=None, policy_grid=None):
        self.value_grid = value_grid
        self.policy_grid = policy_grid

    def surface(self):
        return pygame.display.set_mode((self.width, self.height))

    def draw(self, surface, info_lines=None):
        surface.fill(self.COLORS["bg"])
        self._draw_grid(surface)
        self._draw_cells(surface)
        self._draw_agent(surface)
        self._draw_overlays(surface)
        self._draw_side_panel(surface, info_lines or [])

    def _rc_to_rect(self, r, c):
        x = self.margin + c * self.cell_size
        y = self.margin + r * self.cell_size
        return pygame.Rect(x, y, self.cell_size, self.cell_size)

    def _draw_grid(self, surface):
        for r in range(self.rows):
            for c in range(self.cols):
                rect = self._rc_to_rect(r, c)
                pygame.draw.rect(surface, self.COLORS["grid"], rect, width=1)

    def _draw_cells(self, surface):
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.env.original_grid[r][c]
                rect = self._rc_to_rect(r, c)
                if cell == "#":
                    pygame.draw.rect(surface, self.COLORS["wall"], rect)
                elif cell == "S":
                    pygame.draw.rect(surface, self._with_alpha(self.COLORS["start"], 90), rect)
                    self._draw_center_text(surface, rect, "S", self.font_md)
                elif cell == "G":
                    pygame.draw.rect(surface, self._with_alpha(self.COLORS["goal"], 90), rect)
                    self._draw_center_text(surface, rect, "G", self.font_md)
                elif cell == "K":
                    pygame.draw.rect(surface, self._with_alpha(self.COLORS["key"], 90), rect)
                    self._draw_center_text(surface, rect, "K", self.font_md)
                elif cell == "P":
                    pygame.draw.rect(surface, self._with_alpha(self.COLORS["penalty"], 70), rect)
                    self._draw_center_text(surface, rect, "!", self.font_md)
                elif cell == "E":
                    pygame.draw.rect(surface, self._with_alpha(self.COLORS["energy"], 60), rect)
                    self._draw_center_text(surface, rect, "E", self.font_md)

    def _draw_agent(self, surface):
        r, c = self.env.agent_pos
        rect = self._rc_to_rect(r, c)
        pad = max(3, self.cell_size // 8)
        rect = rect.inflate(-2 * pad, -2 * pad)
        pygame.draw.rect(surface, self.COLORS["agent"], rect, border_radius=6)

    def _draw_overlays(self, surface):
        # Value heatmap
        if self.value_grid is not None:
            v = self.value_grid.astype(float)
            v_mask = ~np.isnan(v)
            if np.any(v_mask):
                vmin = float(np.nanmin(v))
                vmax = float(np.nanmax(v))
                if math.isclose(vmin, vmax):
                    vmax = vmin + 1e-6
                for r in range(self.rows):
                    for c in range(self.cols):
                        if not v_mask[r, c]:
                            continue
                        rect = self._rc_to_rect(r, c)
                        color = self._value_to_color(float(v[r, c]), vmin, vmax)
                        overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                        overlay.fill((*color, self.overlay_alpha))
                        surface.blit(overlay, (rect.x, rect.y))

        # Policy arrows
        if self.policy_grid is not None:
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.env.original_grid[r][c] == "#":
                        continue
                    a = int(self.policy_grid[r, c])
                    if a < 0:
                        continue
                    rect = self._rc_to_rect(r, c)
                    self._draw_arrow(surface, rect, a, color=(30, 220, 255))

    def _with_alpha(self, rgb, alpha):
        r, g, b = rgb
        factor = alpha / 255.0
        return (int(r * factor), int(g * factor), int(b * factor))

    def _value_to_color(self, value, vmin, vmax):
        # Map value to color gradient: red -> yellow -> green
        t = (value - vmin) / max(1e-9, (vmax - vmin))
        t = max(0.0, min(1.0, t))
        if t < 0.5:
            # red (255,60,60) to yellow (255,255,100)
            k = t / 0.5
            r = 255
            g = int(60 + k * (255 - 60))
            b = int(60 + k * (100 - 60))
        else:
            # yellow (255,255,100) to green (50,220,70)
            k = (t - 0.5) / 0.5
            r = int(255 + k * (50 - 255))
            g = int(255 + k * (220 - 255))
            b = int(100 + k * (70 - 100))
        return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))

    def _draw_arrow(self, surface, rect, action, color=(255, 255, 255)):
        cx = rect.x + rect.w // 2
        cy = rect.y + rect.h // 2
        length = max(6, rect.w // 3)
        head = max(4, rect.w // 7)
        if action == 0:  # up
            end = (cx, cy - length)
            left = (cx - head, cy - length + head)
            right = (cx + head, cy - length + head)
        elif action == 1:  # down
            end = (cx, cy + length)
            left = (cx - head, cy + length - head)
            right = (cx + head, cy + length - head)
        elif action == 2:  # left
            end = (cx - length, cy)
            left = (cx - length + head, cy - head)
            right = (cx - length + head, cy + head)
        else:  # right
            end = (cx + length, cy)
            left = (cx + length - head, cy - head)
            right = (cx + length - head, cy + head)
        pygame.draw.line(surface, color, (cx, cy), end, width=2)
        pygame.draw.polygon(surface, color, [end, left, right])

    def _draw_center_text(self, surface, rect, text, font, color=None):
        color = color or self.COLORS["text"]
        img = font.render(str(text), True, color)
        x = rect.x + (rect.w - img.get_width()) // 2
        y = rect.y + (rect.h - img.get_height()) // 2
        surface.blit(img, (x, y))

    def _draw_side_panel(self, surface, info_lines):
        panel_rect = pygame.Rect(
            self.margin + self.cols * self.cell_size, self.margin, self.side_panel, self.rows * self.cell_size
        )
        pygame.draw.rect(surface, self.COLORS["panel"], panel_rect)

        pad = 10
        x = panel_rect.x + pad
        y = panel_rect.y + pad

        title = self.font_lg.render("RL Maze", True, self.COLORS["accent"])
        surface.blit(title, (x, y))
        y += title.get_height() + 8

        # Instructions
        instructions = [
            "[1] Value Iteration   [2] Q-Learning   [3] SARSA(λ)",
            "[Space] Train/Pause   [N] Step batch   [R] Reset",
            "[V] Reward: shaped/sparse   [H] Heatmap   [P] Policy",
            "[E] Evaluate policy   [S] Save screenshot",
            "[T] Transfer demo     [+/-] Speed",
        ]
        for line in instructions:
            img = self.font_sm.render(line, True, self.COLORS["text"])
            surface.blit(img, (x, y))
            y += img.get_height() + 2

        y += 8
        # Dynamic info
        for line in info_lines:
            img = self.font_md.render(line, True, self.COLORS["text"])
            surface.blit(img, (x, y))
            y += img.get_height() + 4
