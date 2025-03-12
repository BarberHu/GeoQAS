const fs = require('fs');
const path = require('path');

// Magic 配置
const config = {
  mcpServers: {
    'github.com/21st-dev/magic-mcp': {
      command: 'C:\\Windows\\System32\\cmd.exe',
      args: [
        '/c',
        'npx',
        '@smithery/cli@latest',
        'run',
        '@21st-dev/magic-mcp',
        '--config',
        JSON.stringify({
          // 用户的API密钥
          TWENTY_FIRST_API_KEY: 'd200115a8fdeb797ebe49b9cf8633695cacdc8355f9f135b027c1f937f53c53c',
          API_URL: 'https://api.21st.dev',
          DEBUG: 'true'
        })
      ],
      disabled: false,
      autoApprove: [],
      env: {
        NODE_ENV: 'production',
        // 用户的API密钥
        TWENTY_FIRST_API_KEY: 'd200115a8fdeb797ebe49b9cf8633695cacdc8355f9f135b027c1f937f53c53c',
        NODE_OPTIONS: '--no-warnings',
        NPM_CONFIG_UPDATE_NOTIFIER: 'false',
        DEBUG: 'true',
        API_URL: 'https://api.21st.dev'
      }
    }
  }
};

// 配置文件路径
const configPath = path.join(process.env.APPDATA, 'Cursor', 'User', 'globalStorage', 'saoudrizwan.claude-dev', 'settings', 'cline_mcp_settings.json');

try {
  // 确保目录存在
  const configDir = path.dirname(configPath);
  if (!fs.existsSync(configDir)) {
    fs.mkdirSync(configDir, { recursive: true });
  }
  
  // 写入配置文件
  fs.writeFileSync(configPath, JSON.stringify(config, null, 2));
  console.log('配置文件已成功创建:', configPath);
  
  // 输出完整配置以供检查
  console.log('\n当前配置:');
  console.log(JSON.stringify(config, null, 2));
} catch (error) {
  console.error('创建配置文件时出错:', error);
} 