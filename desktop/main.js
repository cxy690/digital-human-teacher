"""AI 教师课堂桌面客户端（Electron）。

设计要点：学生端页面里所有地址都是相对路径（fetch("/api/...")、WebSocket 用
location.host），一旦用 file:// 加载本地页面就会全部失效。因此本客户端
**加载远程课堂页面**，让浏览器按同源规则自动解析地址 —— 前端代码零改动，
且云端更新页面后客户端自动生效。

服务器地址写在同目录 config.json；为空（或首次运行）时弹出设置页让用户填写。
"""
const { app, BrowserWindow, ipcMain, shell } = require("electron");
const fs = require("fs");
const path = require("path");

// 配置文件：开发时放 desktop 目录；打包后放到 exe 同目录（用户可改）
function configPath() {
  if (app.isPackaged) {
    return path.join(path.dirname(app.getPath("exe")), "config.json");
  }
  return path.join(__dirname, "config.json");
}

function readConfig() {
  try {
    return JSON.parse(fs.readFileSync(configPath(), "utf-8"));
  } catch (e) {
    return {};
  }
}

function writeConfig(cfg) {
  try {
    fs.writeFileSync(configPath(), JSON.stringify(cfg, null, 2), "utf-8");
  } catch (e) {
    /* 只读目录下写入失败不影响本次运行 */
  }
}

let mainWindow = null;
let setupWindow = null;

function openClassroom(serverUrl) {
  if (setupWindow) { setupWindow.close(); setupWindow = null; }
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    title: "AI 教师课堂",
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      // 允许页面自动播放音视频（数字人讲解需要）
      autoplayPolicy: "no-user-gesture-required",
    },
  });
  mainWindow.setMenuBarVisibility(false);
  // 学科选择页：与浏览器里访问的地址一致
  mainWindow.loadURL(serverUrl + "/app/subject_select/");
  mainWindow.on("closed", () => { mainWindow = null; });
  // 外链用系统浏览器打开，避免客户端被导航走
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith(serverUrl)) return { action: "allow" };
    shell.openExternal(url);
    return { action: "deny" };
  });
}

function openSetup() {
  setupWindow = new BrowserWindow({
    width: 560,
    height: 420,
    title: "连接设置",
    resizable: false,
    webPreferences: {
      nodeIntegration: true,       // 配置页需要 ipcRenderer
      contextIsolation: false,
    },
  });
  setupWindow.setMenuBarVisibility(false);
  setupWindow.loadFile(path.join(__dirname, "setup.html"));
  setupWindow.on("closed", () => {
    setupWindow = null;
    // 设置页被关闭且还没进入课堂 -> 退出，避免留下空白窗口
    if (!mainWindow) app.quit();
  });
}

ipcMain.handle("get-server-url", () => (readConfig().serverUrl || "").trim());
ipcMain.handle("set-server-url", (_e, url) => {
  const cfg = readConfig();
  cfg.serverUrl = url;
  writeConfig(cfg);
  openClassroom(url);
});

app.whenReady().then(() => {
  const url = (readConfig().serverUrl || "").trim();
  if (url) openClassroom(url);
  else openSetup();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      const u = (readConfig().serverUrl || "").trim();
      if (u) openClassroom(u); else openSetup();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
