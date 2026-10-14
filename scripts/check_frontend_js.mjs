// 前端内联 JS 语法体检（免依赖，只需 Node 18+）
//
// 用途：clients/*/index.html 的 JS 都是内联的 <script>，改动（尤其注释掉大段代码）
// 很容易写出语法错误却要等浏览器打开才发现。本脚本不开浏览器，直接做语法解析。
//
// 用法：
//   node scripts/check_frontend_js.mjs clients/student_tablet/index.html
//   node scripts/check_frontend_js.mjs clients/subject_select/index.html
//
// 检查项：
//   1) 每段 <script> 能否被 JS 引擎解析（new Function 编译，不执行）
//   2) HTML 注释 <!-- 与 --> 是否配对（注释写坏会把整块页面结构吞掉）
// 退出码：0 = 通过，1 = 有问题（可用于提交前自检）
import fs from "node:fs";

const files = process.argv.slice(2);
if (!files.length) {
  console.error("usage: node scripts/check_frontend_js.mjs <html-file> [...]");
  process.exit(2);
}

let failed = false;
for (const file of files) {
  let html;
  try {
    html = fs.readFileSync(file, "utf8");
  } catch (e) {
    console.log(`[MISS] ${file}: ${e.message}`);
    failed = true;
    continue;
  }

  const blocks = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];
  let bad = 0;
  const results = blocks.map((b, i) => {
    try {
      new Function(b[1]);          // 只编译不执行
      return `    script[${i}] OK (${b[1].split("\n").length} lines)`;
    } catch (e) {
      bad += 1;
      return `    script[${i}] SYNTAX ERROR -> ${e.message}`;
    }
  });

  const opens = (html.match(/<!--/g) || []).length;
  const closes = (html.match(/-->/g) || []).length;
  const commentsOk = opens === closes;

  const ok = bad === 0 && commentsOk;
  if (!ok) failed = true;
  console.log(`${ok ? "[ OK ]" : "[FAIL]"} ${file}`);
  console.log(`    script blocks: ${blocks.length}`);
  results.forEach((r) => console.log(r));
  console.log(`    html comments: open=${opens} close=${closes} ${commentsOk ? "OK" : "MISMATCH"}`);
}

process.exit(failed ? 1 : 0);
