function startBot(botId) {
    fetch(`/start_bot`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bot_id: botId })
    });
}

function stopBot(botId) {
    fetch(`/stop_bot`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bot_id: botId })
    });
}

function pauseBot(botId) {
    fetch(`/pause_bot`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bot_id: botId })
    });
}

function generateReport() {
    fetch(`/generate_report`, {
        method: 'GET'
    }).then(response => response.json()).then(data => {
        document.getElementById('report-output').innerHTML = JSON.stringify(data, null, 2);
    });
}
