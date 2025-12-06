#define IMAGE_VIEWER_EXPORTS 
#include <iostream>
#include "SDL3/SDL.h"
#include "SDL3_image/SDL_image.h"
#include "image_viewer.hpp"

struct WindowContext {
    SDL_Window* window;
    SDL_Renderer* renderer;
    SDL_Texture* texture;
};
int renderImage(const std::vector<std::string>& paths,int width, int height){
     if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL init failed: " << SDL_GetError() << std::endl;
        return 1;
    }
    std::vector<WindowContext> windows;

    for (auto& path : paths) {
        WindowContext ctx{};
        ctx.window = SDL_CreateWindow(path.c_str(), width, height, SDL_WINDOW_RESIZABLE);
        ctx.renderer = SDL_CreateRenderer(ctx.window, nullptr);
        ctx.texture = IMG_LoadTexture(ctx.renderer, path.c_str());
        windows.push_back(ctx);
    }

    bool running = true;
    SDL_Event event;

    while (running && !windows.empty()) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) {
                running = false;
            } else if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) {
               // 找到对应窗口并销毁
                for (auto it = windows.begin(); it != windows.end(); ++it) {
                    if (SDL_GetWindowID(it->window) == event.window.windowID) {
                        SDL_DestroyTexture(it->texture);
                        SDL_DestroyRenderer(it->renderer);
                        SDL_DestroyWindow(it->window);
                        windows.erase(it);
                        break;
                    }
                }
            }
        }
        // 渲染所有窗口
        for (auto& win : windows) {
            SDL_SetRenderDrawColor(win.renderer, 0, 0, 0, 255);
            SDL_RenderClear(win.renderer);
            SDL_RenderTexture(win.renderer, win.texture, nullptr, nullptr);
            SDL_RenderPresent(win.renderer);
        }

        SDL_Delay(16); // 控制渲染帧率
    }

    // 清理剩余窗口
    for (auto& win : windows) {
        SDL_DestroyTexture(win.texture);
        SDL_DestroyRenderer(win.renderer);
        SDL_DestroyWindow(win.window);
    }

    SDL_Quit();
    return 0;
}
