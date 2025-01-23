function saveScore(pName, qNo, score) {
    // Create a unique key for each player and question number
    const key = `${pName}_q${qNo}`;
    // Save the score to local storage
    localStorage.setItem(key, score);
}

function saveScore(pName, qNo, score) {
    // Create a unique key for each player and question number
    const key = `${pName}_q${qNo}`;
    // Save the score to local storage
    localStorage.setItem(key, score);
}

function retrieveScore(pName, qNo) {
    // Create a unique key for each player and question number
    const key = `${pName}_q${qNo}`;
    // Retrieve the score from local storage
    return localStorage.getItem(key);
}

function createSummaryTable() {
    const players = ['Player1', 'Player2', 'Player3'];
    const questions = Array.from({ length: 16 }, (_, i) => i + 1);
    let table = '<table border="1"><tr><th>Player</th>';

    // Create table headers for questions
    questions.forEach(qNo => {
        table += `<th>Q${qNo}</th>`;
    });
    table += '</tr>';

    // Create table rows for each player
    players.forEach(player => {
        table += `<tr><td>${player}</td>`;
        questions.forEach(qNo => {
            const score = retrieveScore(player, qNo) || 'N/A';
            table += `<td>${score}</td>`;
        });
        table += '</tr>';
    });

    table += '</table>';
    return table;
}

// Example usage:
const playerName = 'Player1';
const questionNumber = 1;
const playerScore = 100;

// Save the score
saveScore(playerName, questionNumber, playerScore);

// To retrieve the score later
const savedScore = retrieveScore(playerName, questionNumber);
console.log(`Saved score for ${playerName} on question ${questionNumber}: ${savedScore}`);

// Create and display the summary table
document.body.innerHTML = createSummaryTable();