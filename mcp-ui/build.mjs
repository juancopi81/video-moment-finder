import {build} from 'esbuild';
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';

const root = fileURLToPath(new URL('.', import.meta.url));
const output = new URL('../src/api/assets/workspace.html', import.meta.url);
// Emit escaped strings rather than raw multiline SDK templates in the tracked HTML.
const result = await build({absWorkingDir: root, entryPoints: ['src/app.js'], outfile: 'workspace.js', bundle: true, format: 'esm', target: 'es2022', supported: {'template-literal': false}, minify: true, write: false, legalComments: 'inline', define: {'process.env.NODE_ENV': '"production"'}});
const js = result.outputFiles.find(f => f.path.endsWith('.js')).text.replaceAll('</script', '<\\/script');
const css = result.outputFiles.find(f => f.path.endsWith('.css')).text;
const shell = await readFile(new URL('src/shell.html', import.meta.url), 'utf8');
const packages = ['@modelcontextprotocol/ext-apps', '@modelcontextprotocol/sdk', 'zod', '@standard-schema/spec'];
const notices = (await Promise.all(packages.map(async name => `${name}\n${await readFile(new URL(`node_modules/${name}/LICENSE`, import.meta.url), 'utf8')}`))).join('\n\n');
const escapedNotices = notices.replace(/[ \t]+$/gm, '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
// Replacement callbacks preserve literal $& and other JavaScript/CSS strings.
const html = shell.replace('/*VMF_CSS*/', () => css).replace('/*VMF_JS*/', () => js)
  .replace('<!--VMF_NOTICES-->', () => `<template id="vmf-third-party-notices"><pre>${escapedNotices}</pre></template>`);
if (process.argv.includes('--check')) {
  if (await readFile(output, 'utf8') !== html) throw new Error('Workspace bundle is stale. Run npm run build in mcp-ui.');
  console.log('Workspace bundle matches source.');
} else {
  await mkdir(new URL('../src/api/assets/', import.meta.url), {recursive: true});
  await writeFile(output, html);
  console.log(`Bundled workspace: ${Buffer.byteLength(html)} bytes.`);
}
