#!/usr/bin/env node
import fs from 'fs';

const args = process.argv.slice(2);
const prefix = args[0] || '';

let input = '';
for await (const chunk of process.stdin) {
  input += chunk;
}
const data = JSON.parse(input);
const out = { echo: `${prefix}${data.text}` };
process.stdout.write(JSON.stringify(out));
