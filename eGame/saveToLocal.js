function collectScore() {
    saveScore(pName[0], qNo, p1S[qNo]);
    saveScore(pName[1], qNo, p2S[qNo]);
    saveScore(pName[2], qNo, p3S[qNo]);
}

function saveScore(pName, qNo, score) {
    // Create a unique key for each player and question number
    const key = `${pName}_q${qNo}`;
    // Save the score to local storage
    localStorage.setItem(key, score);
    // Save the timestamp to local storage
    const timestampKey = `${pName}_q${qNo}_timestamp`;
    const timestamp = new Date().toLocaleString();
    localStorage.setItem(timestampKey, timestamp);
    //console.log(`Saved score for ${pName} on question ${qNo}: ${score} at ${timestamp}`);
}

function retrieveScore(pName, qNo) {
    // Create a unique key for each player and question number
    const key = `${pName}_q${qNo}`;
    // Retrieve the score from local storage
    return localStorage.getItem(key);
}

function retrieveTimestamp(pName, qNo) {
    // Create a unique key for each player and question number
    const key = `${pName}_q${qNo}_timestamp`;
    // Retrieve the timestamp from local storage
    return localStorage.getItem(key);
}

function createSummaryTable() {
    const players = [pName[0], pName[1], pName[2]];
    const questions = Array.from({ length: 16 }, (_, i) => i + 1);
    let table = '<table border="1"><tr><th>Player</th>';

    // Create table headers for questions
    questions.forEach(qNo => {
        table += `<th>Q${qNo}</th>`;
    });
    table += '<th>Total Score</th><th>Last Saved</th></tr>';

    // Create table rows for each player
    players.forEach(player => {
        let totalScore = 0;
        let lastSaved = '';
        table += `<tr><td>${player}</td>`;
        questions.forEach(qNo => {
            const score = parseInt(retrieveScore(player, qNo)) || 0;
            totalScore += score;
            const timestamp = retrieveTimestamp(player, qNo) || 'N/A';
            if (timestamp !== 'N/A') {
                lastSaved = timestamp;
            }
            table += `<td>${score}</td>`;
        });
        table += `<td>${totalScore}</td><td>${lastSaved}</td></tr>`;
    });

    table += '</table>';
    return table;
}

function deleteSavedData() {
    const players = [pName[0], pName[1], pName[2]];
    const questions = Array.from({ length: 16 }, (_, i) => i + 1);

    players.forEach(player => {
        questions.forEach(qNo => {
            const key = `${player}_q${qNo}`;
            localStorage.removeItem(key);
        });
    });
    //console.log('All saved data has been deleted.');
}

// Example usage:
//const playerName = 'Player1';
//const questionNumber = 1;
//const playerScore = 100;

// Save the score
//saveScore(playerName, questionNumber, playerScore);

// To retrieve the score later
//const savedScore = retrieveScore(playerName, questionNumber);
//console.log(`Saved score for ${playerName} on question ${questionNumber}: ${savedScore}`);

// Create and display the summary table
//document.body.innerHTML = createSummaryTable();