import pygame


class Camera:
    def __init__(self, width, height):
        self.position = pygame.Vector2(width/2,height/2)
        self.zoom = 1.0
        self.drag_start = None
        self.size = pygame.Rect(0,0,width,height)
        self.updateFov()

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4:  # Scroll wheel up
                self.zoom *= 1.1
            elif event.button == 5:  # Scroll wheel down
                self.zoom /= 1.1
            elif event.button == 1:  # Left mouse button
                self.drag_start = pygame.mouse.get_pos()

            if(self.zoom<1):
                self.zoom=1.
            elif(self.zoom>20):
                self.zoom=20

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                self.drag_start = None
        elif event.type == pygame.MOUSEMOTION:
            if self.drag_start is not None:
                x, y = event.pos
                x0, y0 = self.drag_start
                self.position.x += (x0 - x) / self.zoom
                self.position.y += (y0 - y) / self.zoom
                self.drag_start = event.pos
            

        self.constrainCam()
        self.updateFov()

    def constrainCam(self):
        fsize = pygame.Vector2(self.fov.size)/2
        minx,miny = fsize
        maxx,maxy = pygame.Vector2(self.size.size)-fsize

        self.position.x = max(min(self.position.x,maxx),minx)
        self.position.y = max(min(self.position.y,maxy),miny)

    def updateFov(self):
        self.fov=pygame.Rect(0,0,int(self.size.w/self.zoom),int(self.size.h/self.zoom))
        self.fov.center = self.position
    
    def apply(self, surface):
        visible_surface = pygame.Surface((self.fov.w, self.fov.h))

        visible_surface.blit(surface, (0,0), self.fov)

        return pygame.transform.scale(visible_surface, (self.size.w, self.size.h))
