/**
 * Sports Betting Predictor Dashboard - JavaScript
 * Fetches live data from the API
 */

const API_BASE = '';  // Same origin

// State
let currentSport = 'nba';
let lastUpdate = null;

// Team data for all sports (abbr + full name)
const TEAMS = {
    nba: [
        { abbr: 'ATL', name: 'Atlanta Hawks' }, { abbr: 'BOS', name: 'Boston Celtics' },
        { abbr: 'BKN', name: 'Brooklyn Nets' }, { abbr: 'CHA', name: 'Charlotte Hornets' },
        { abbr: 'CHI', name: 'Chicago Bulls' }, { abbr: 'CLE', name: 'Cleveland Cavaliers' },
        { abbr: 'DAL', name: 'Dallas Mavericks' }, { abbr: 'DEN', name: 'Denver Nuggets' },
        { abbr: 'DET', name: 'Detroit Pistons' }, { abbr: 'GSW', name: 'Golden State Warriors' },
        { abbr: 'HOU', name: 'Houston Rockets' }, { abbr: 'IND', name: 'Indiana Pacers' },
        { abbr: 'LAC', name: 'LA Clippers' }, { abbr: 'LAL', name: 'Los Angeles Lakers' },
        { abbr: 'MEM', name: 'Memphis Grizzlies' }, { abbr: 'MIA', name: 'Miami Heat' },
        { abbr: 'MIL', name: 'Milwaukee Bucks' }, { abbr: 'MIN', name: 'Minnesota Timberwolves' },
        { abbr: 'NOP', name: 'New Orleans Pelicans' }, { abbr: 'NYK', name: 'New York Knicks' },
        { abbr: 'OKC', name: 'Oklahoma City Thunder' }, { abbr: 'ORL', name: 'Orlando Magic' },
        { abbr: 'PHI', name: 'Philadelphia 76ers' }, { abbr: 'PHX', name: 'Phoenix Suns' },
        { abbr: 'POR', name: 'Portland Trail Blazers' }, { abbr: 'SAC', name: 'Sacramento Kings' },
        { abbr: 'SAS', name: 'San Antonio Spurs' }, { abbr: 'TOR', name: 'Toronto Raptors' },
        { abbr: 'UTA', name: 'Utah Jazz' }, { abbr: 'WAS', name: 'Washington Wizards' }
    ],
    nfl: [
        { abbr: 'ARI', name: 'Arizona Cardinals' }, { abbr: 'ATL', name: 'Atlanta Falcons' },
        { abbr: 'BAL', name: 'Baltimore Ravens' }, { abbr: 'BUF', name: 'Buffalo Bills' },
        { abbr: 'CAR', name: 'Carolina Panthers' }, { abbr: 'CHI', name: 'Chicago Bears' },
        { abbr: 'CIN', name: 'Cincinnati Bengals' }, { abbr: 'CLE', name: 'Cleveland Browns' },
        { abbr: 'DAL', name: 'Dallas Cowboys' }, { abbr: 'DEN', name: 'Denver Broncos' },
        { abbr: 'DET', name: 'Detroit Lions' }, { abbr: 'GB', name: 'Green Bay Packers' },
        { abbr: 'HOU', name: 'Houston Texans' }, { abbr: 'IND', name: 'Indianapolis Colts' },
        { abbr: 'JAX', name: 'Jacksonville Jaguars' }, { abbr: 'KC', name: 'Kansas City Chiefs' },
        { abbr: 'KAN', name: 'Kansas City Chiefs' }, { abbr: 'LAC', name: 'Los Angeles Chargers' },
        { abbr: 'LAR', name: 'Los Angeles Rams' }, { abbr: 'LV', name: 'Las Vegas Raiders' },
        { abbr: 'MIA', name: 'Miami Dolphins' }, { abbr: 'MIN', name: 'Minnesota Vikings' },
        { abbr: 'NE', name: 'New England Patriots' }, { abbr: 'NEW', name: 'New England Patriots' },
        { abbr: 'NO', name: 'New Orleans Saints' }, { abbr: 'NYG', name: 'New York Giants' },
        { abbr: 'NYJ', name: 'New York Jets' }, { abbr: 'PHI', name: 'Philadelphia Eagles' },
        { abbr: 'PIT', name: 'Pittsburgh Steelers' }, { abbr: 'SEA', name: 'Seattle Seahawks' },
        { abbr: 'SF', name: 'San Francisco 49ers' }, { abbr: 'TB', name: 'Tampa Bay Buccaneers' },
        { abbr: 'TEN', name: 'Tennessee Titans' }, { abbr: 'WAS', name: 'Washington Commanders' },
        { abbr: 'WSH', name: 'Washington Commanders' }
    ],
    mlb: [
        { abbr: 'ARI', name: 'Arizona Diamondbacks' }, { abbr: 'ATL', name: 'Atlanta Braves' },
        { abbr: 'BAL', name: 'Baltimore Orioles' }, { abbr: 'BOS', name: 'Boston Red Sox' },
        { abbr: 'CHC', name: 'Chicago Cubs' }, { abbr: 'CHW', name: 'Chicago White Sox' },
        { abbr: 'CIN', name: 'Cincinnati Reds' }, { abbr: 'CLE', name: 'Cleveland Guardians' },
        { abbr: 'COL', name: 'Colorado Rockies' }, { abbr: 'DET', name: 'Detroit Tigers' },
        { abbr: 'HOU', name: 'Houston Astros' }, { abbr: 'KC', name: 'Kansas City Royals' },
        { abbr: 'LAA', name: 'Los Angeles Angels' }, { abbr: 'LAD', name: 'Los Angeles Dodgers' },
        { abbr: 'MIA', name: 'Miami Marlins' }, { abbr: 'MIL', name: 'Milwaukee Brewers' },
        { abbr: 'MIN', name: 'Minnesota Twins' }, { abbr: 'NYM', name: 'New York Mets' },
        { abbr: 'NYY', name: 'New York Yankees' }, { abbr: 'OAK', name: 'Oakland Athletics' },
        { abbr: 'PHI', name: 'Philadelphia Phillies' }, { abbr: 'PIT', name: 'Pittsburgh Pirates' },
        { abbr: 'SD', name: 'San Diego Padres' }, { abbr: 'SF', name: 'San Francisco Giants' },
        { abbr: 'SEA', name: 'Seattle Mariners' }, { abbr: 'STL', name: 'St. Louis Cardinals' },
        { abbr: 'TB', name: 'Tampa Bay Rays' }, { abbr: 'TEX', name: 'Texas Rangers' },
        { abbr: 'TOR', name: 'Toronto Blue Jays' }, { abbr: 'WAS', name: 'Washington Nationals' }
    ],
    nhl: [
        { abbr: 'ANA', name: 'Anaheim Ducks' }, { abbr: 'ARI', name: 'Arizona Coyotes' },
        { abbr: 'BOS', name: 'Boston Bruins' }, { abbr: 'BUF', name: 'Buffalo Sabres' },
        { abbr: 'CAR', name: 'Carolina Hurricanes' }, { abbr: 'CBJ', name: 'Columbus Blue Jackets' },
        { abbr: 'CGY', name: 'Calgary Flames' }, { abbr: 'CHI', name: 'Chicago Blackhawks' },
        { abbr: 'COL', name: 'Colorado Avalanche' }, { abbr: 'DAL', name: 'Dallas Stars' },
        { abbr: 'DET', name: 'Detroit Red Wings' }, { abbr: 'EDM', name: 'Edmonton Oilers' },
        { abbr: 'FLA', name: 'Florida Panthers' }, { abbr: 'LA', name: 'Los Angeles Kings' },
        { abbr: 'MIN', name: 'Minnesota Wild' }, { abbr: 'MTL', name: 'Montreal Canadiens' },
        { abbr: 'NJ', name: 'New Jersey Devils' }, { abbr: 'NSH', name: 'Nashville Predators' },
        { abbr: 'NYI', name: 'New York Islanders' }, { abbr: 'NYR', name: 'New York Rangers' },
        { abbr: 'OTT', name: 'Ottawa Senators' }, { abbr: 'PHI', name: 'Philadelphia Flyers' },
        { abbr: 'PIT', name: 'Pittsburgh Penguins' }, { abbr: 'SEA', name: 'Seattle Kraken' },
        { abbr: 'SJ', name: 'San Jose Sharks' }, { abbr: 'STL', name: 'St. Louis Blues' },
        { abbr: 'TB', name: 'Tampa Bay Lightning' }, { abbr: 'TOR', name: 'Toronto Maple Leafs' },
        { abbr: 'VAN', name: 'Vancouver Canucks' }, { abbr: 'VGK', name: 'Vegas Golden Knights' },
        { abbr: 'WPG', name: 'Winnipeg Jets' }, { abbr: 'WSH', name: 'Washington Capitals' }
    ],
    ncaaf: [
        { abbr: 'ALA', name: 'Alabama Crimson Tide' }, { abbr: 'ARZ', name: 'Arizona Wildcats' },
        { abbr: 'ASU', name: 'Arizona State Sun Devils' }, { abbr: 'ARK', name: 'Arkansas Razorbacks' },
        { abbr: 'AUB', name: 'Auburn Tigers' }, { abbr: 'BAY', name: 'Baylor Bears' },
        { abbr: 'CAL', name: 'California Golden Bears' }, { abbr: 'CLE', name: 'Clemson Tigers' },
        { abbr: 'COL', name: 'Colorado Buffaloes' }, { abbr: 'DUK', name: 'Duke Blue Devils' },
        { abbr: 'FLA', name: 'Florida Gators' }, { abbr: 'FSU', name: 'Florida State Seminoles' },
        { abbr: 'UGA', name: 'Georgia Bulldogs' }, { abbr: 'ILL', name: 'Illinois Fighting Illini' },
        { abbr: 'IND', name: 'Indiana Hoosiers' }, { abbr: 'IOW', name: 'Iowa Hawkeyes' },
        { abbr: 'ISU', name: 'Iowa State Cyclones' }, { abbr: 'KAN', name: 'Kansas Jayhawks' },
        { abbr: 'KSU', name: 'Kansas State Wildcats' }, { abbr: 'KEN', name: 'Kentucky Wildcats' },
        { abbr: 'LSU', name: 'LSU Tigers' }, { abbr: 'LOU', name: 'Louisville Cardinals' },
        { abbr: 'UMD', name: 'Maryland Terrapins' }, { abbr: 'MIA', name: 'Miami Hurricanes' },
        { abbr: 'MICH', name: 'Michigan Wolverines' }, { abbr: 'MSU', name: 'Michigan State Spartans' },
        { abbr: 'MINN', name: 'Minnesota Golden Gophers' }, { abbr: 'MISS', name: 'Ole Miss Rebels' },
        { abbr: 'MSST', name: 'Mississippi State Bulldogs' }, { abbr: 'MIZ', name: 'Missouri Tigers' },
        { abbr: 'NEB', name: 'Nebraska Cornhuskers' }, { abbr: 'NC', name: 'North Carolina Tar Heels' },
        { abbr: 'NCST', name: 'NC State Wolfpack' }, { abbr: 'ND', name: 'Notre Dame Fighting Irish' },
        { abbr: 'OSU', name: 'Ohio State Buckeyes' }, { abbr: 'OU', name: 'Oklahoma Sooners' },
        { abbr: 'OKST', name: 'Oklahoma State Cowboys' }, { abbr: 'ORE', name: 'Oregon Ducks' },
        { abbr: 'ORST', name: 'Oregon State Beavers' }, { abbr: 'PSU', name: 'Penn State Nittany Lions' },
        { abbr: 'PITT', name: 'Pittsburgh Panthers' }, { abbr: 'PUR', name: 'Purdue Boilermakers' },
        { abbr: 'RUT', name: 'Rutgers Scarlet Knights' }, { abbr: 'USC', name: 'USC Trojans' },
        { abbr: 'SCAR', name: 'South Carolina Gamecocks' }, { abbr: 'STAN', name: 'Stanford Cardinal' },
        { abbr: 'SYR', name: 'Syracuse Orange' }, { abbr: 'TENN', name: 'Tennessee Volunteers' },
        { abbr: 'TEX', name: 'Texas Longhorns' }, { abbr: 'TAMU', name: 'Texas A&M Aggies' },
        { abbr: 'TT', name: 'Texas Tech Red Raiders' }, { abbr: 'TCU', name: 'TCU Horned Frogs' },
        { abbr: 'UCLA', name: 'UCLA Bruins' }, { abbr: 'UTAH', name: 'Utah Utes' },
        { abbr: 'UVA', name: 'Virginia Cavaliers' }, { abbr: 'VT', name: 'Virginia Tech Hokies' },
        { abbr: 'WAKE', name: 'Wake Forest Demon Deacons' }, { abbr: 'WASH', name: 'Washington Huskies' },
        { abbr: 'WSU', name: 'Washington State Cougars' }, { abbr: 'WVU', name: 'West Virginia Mountaineers' },
        { abbr: 'WIS', name: 'Wisconsin Badgers' }
    ]
};

// Tab Navigation
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        const tabId = btn.dataset.tab;
        document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
        document.getElementById(tabId).classList.add('active');

        // Load data for tab
        if (tabId === 'predictions') loadPredictions();
        if (tabId === 'edge') loadEdgeAnalysis();
        if (tabId === 'performance') loadPerformance();
    });
});

// ============== Data Loading ==============

// Store all bets for filtering
let allBets = [];

async function loadAllBets() {
    const container = document.getElementById('bets-container');
    container.innerHTML = '<div class="loading"></div>';

    const sports = ['nba', 'nfl', 'nhl', 'ncaaf'];
    allBets = [];

    // Fetch moneyline edges
    for (const sport of sports) {
        try {
            const response = await fetch(`${API_BASE}/api/edge-analysis?sport=${sport}`);
            const data = await response.json();

            if (data.edges) {
                data.edges.forEach(edge => {
                    allBets.push({
                        ...edge,
                        sport: sport.toUpperCase(),
                        betType: 'moneyline'
                    });
                });
            }
        } catch (e) {
            console.log(`${sport} moneyline error:`, e);
        }
    }

    // Fetch spread edges
    for (const sport of sports) {
        try {
            const response = await fetch(`${API_BASE}/api/spread-analysis?sport=${sport}`);
            const data = await response.json();

            if (data.edges) {
                data.edges.forEach(edge => {
                    allBets.push({
                        ...edge,
                        sport: sport.toUpperCase(),
                        betType: 'spread'
                    });
                });
            }
        } catch (e) {
            console.log(`${sport} spread error:`, e);
        }
    }

    // Fetch totals (over/under) edges
    for (const sport of sports) {
        try {
            const response = await fetch(`${API_BASE}/api/totals-analysis?sport=${sport}`);
            const data = await response.json();

            if (data.edges) {
                data.edges.forEach(edge => {
                    allBets.push({
                        ...edge,
                        sport: sport.toUpperCase(),
                        betType: 'total',
                        team: `${edge.pick} ${edge.line}`  // Display as "OVER 220.5"
                    });
                });
            }
        } catch (e) {
            console.log(`${sport} totals error:`, e);
        }
    }

    // Sort by EV descending
    allBets.sort((a, b) => b.ev - a.ev);

    applyFilters();
}

function applyFilters() {
    const sportFilter = document.getElementById('filter-sport').value;
    const betTypeFilter = document.getElementById('filter-bettype').value;
    const evFilter = parseFloat(document.getElementById('filter-ev').value);

    let filtered = allBets.filter(bet => {
        if (sportFilter !== 'all' && bet.sport.toLowerCase() !== sportFilter) return false;
        if (betTypeFilter !== 'all' && bet.betType !== betTypeFilter) return false;
        if (bet.ev * 100 < evFilter) return false;
        return true;
    });

    // Limit to top 15 best bets (already sorted by EV)
    const totalBeforeLimit = filtered.length;
    filtered = filtered.slice(0, 15);

    // Update summary
    const countText = totalBeforeLimit > 15 ? `15 of ${totalBeforeLimit}` : `${filtered.length}`;
    document.getElementById('total-edges').textContent = countText;
    const avgEv = filtered.length > 0
        ? filtered.reduce((sum, b) => sum + b.ev, 0) / filtered.length
        : 0;
    document.getElementById('avg-ev').textContent = `+${(avgEv * 100).toFixed(1)}%`;
    const uniqueSports = new Set(filtered.map(b => b.sport));
    document.getElementById('total-sports').textContent = uniqueSports.size;

    renderBets(filtered);
}

// ============== Helpers ==============

function getTeamName(abbr, sport) {
    if (!abbr) return '';
    sport = sport.toLowerCase();
    const sportTeams = TEAMS[sport] || [];
    const team = sportTeams.find(t => t.abbr === abbr);
    return team ? team.name : abbr;
}

function getConfidenceClass(confidence) {
    if (confidence >= 0.65) return 'confidence-high';
    if (confidence >= 0.55) return 'confidence-medium';
    return 'confidence-low';
}

// ... existing helpers ...

function renderBets(bets) {
    const container = document.getElementById('bets-container');

    if (bets.length === 0) {
        container.innerHTML = '<div class="no-data">No bets match your filters</div>';
        return;
    }

    container.innerHTML = bets.map((bet, idx) => {
        const fullTeamName = getTeamName(bet.team, bet.sport);
        const spreadInfo = bet.betType === 'spread' && bet.spread !== undefined
            ? ` (${bet.spread > 0 ? '+' : ''}${bet.spread})`
            : '';

        // Determine bet type class
        let betTypeClass = 'bet-type-moneyline';
        if (bet.betType === 'spread') betTypeClass = 'bet-type-spread';
        if (bet.betType === 'total') betTypeClass = 'bet-type-total';

        // Determine probability label
        let probLabel = 'Win Prob';
        if (bet.betType === 'spread') probLabel = 'Cover Prob';
        if (bet.betType === 'total') probLabel = 'Hit Prob';

        // Format game time
        const gameTime = bet.startTime ? formatGameTime(bet.startTime) : '';

        return `
        <div class="bet-card ${bet.ev > 0.05 ? 'strong-edge' : ''}" onclick="showBetDetail(${idx})">
            <div class="bet-card-header">
                <span class="bet-sport-badge">${bet.sport}</span>
                <span class="bet-ev">+${(bet.ev * 100).toFixed(1)}%</span>
            </div>
            <div class="bet-matchup">${bet.game}</div>
            ${gameTime ? `<div class="bet-time">📅 ${gameTime}</div>` : ''}
            <div class="bet-pick">
                <span class="bet-pick-team">${fullTeamName}${spreadInfo}</span>
                <span class="bet-pick-odds">${formatOdds(bet.odds)}</span>
            </div>
            <div class="bet-meta">
                <span>${probLabel}: ${(bet.modelProbability * 100).toFixed(0)}%</span>
                <span class="bet-type-badge ${betTypeClass}">${bet.betType}</span>
            </div>
        </div>
    `}).join('');
}

function showBetDetail(idx) {
    const bet = allBets.filter(b => {
        const sportFilter = document.getElementById('filter-sport').value;
        const betTypeFilter = document.getElementById('filter-bettype').value;
        const evFilter = parseFloat(document.getElementById('filter-ev').value);
        if (sportFilter !== 'all' && b.sport.toLowerCase() !== sportFilter) return false;
        if (betTypeFilter !== 'all' && b.betType !== betTypeFilter) return false;
        if (b.ev * 100 < evFilter) return false;
        return true;
    })[idx];

    if (!bet) return;

    const fullTeamName = getTeamName(bet.team, bet.sport);
    const teams = bet.game.split(' @ ');
    const awayTeam = teams[0] || bet.team;
    const homeTeam = teams[1] || '';

    const googleUrl = `https://www.google.com/search?q=${encodeURIComponent(bet.game + ' ' + bet.sport)}`;
    const stakeUrl = `https://stake.com/sports/${bet.sport.toLowerCase()}`;

    const modal = document.getElementById('bet-modal');
    document.getElementById('modal-body').innerHTML = `
        <div class="modal-header">
            <div class="modal-matchup">${bet.game}</div>
            <div style="color: var(--text-muted); margin-top: 4px;">${bet.sport} | ${bet.betType}</div>
        </div>
        
        <div class="modal-section">
            <h4>Recommended Bet</h4>
            <div class="bet-pick" style="font-size: 1.1rem;">
                <span class="bet-pick-team">${fullTeamName}</span>
                <span class="bet-pick-odds">${formatOdds(bet.odds)}</span>
            </div>
        </div>
        
        <div class="modal-section">
            <h4>Why This Edge?</h4>
            <p style="color: var(--text-secondary); line-height: 1.6;">
                Our model gives <strong>${fullTeamName}</strong> a <strong>${(bet.modelProbability * 100).toFixed(1)}%</strong> 
                chance to win, but the odds imply only ~${(100 / (Math.abs(bet.odds) + (bet.odds > 0 ? 100 : 0)) * (bet.odds > 0 ? 100 : Math.abs(bet.odds))).toFixed(0)}%.
                This creates <strong style="color: var(--success);">+${(bet.ev * 100).toFixed(1)}% expected value</strong>.
            </p>
        </div>
        
        <div class="modal-section">
            <h4>Model Analysis</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                <div style="background: var(--bg-secondary); padding: 12px; border-radius: 8px;">
                    <div style="color: var(--text-muted); font-size: 0.8rem;">Win Probability</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: var(--accent-primary);">${(bet.modelProbability * 100).toFixed(1)}%</div>
                </div>
                <div style="background: var(--bg-secondary); padding: 12px; border-radius: 8px;">
                    <div style="color: var(--text-muted); font-size: 0.8rem;">Expected Value</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: var(--success);">+${(bet.ev * 100).toFixed(1)}%</div>
                </div>
            </div>
        </div>
        
        <div class="modal-section">
            <h4>Quick Links</h4>
            <div class="modal-links">
                <a href="${googleUrl}" target="_blank" class="modal-link">🔍 Google</a>
                <a href="https://www.espn.com/${bet.sport.toLowerCase()}" target="_blank" class="modal-link">📰 ESPN</a>
                <a href="${stakeUrl}" target="_blank" class="modal-link stake">🎰 Stake</a>
            </div>
        </div>
    `;

    modal.classList.add('active');
}

function closeModal(event) {
    if (event && event.target !== event.currentTarget) return;
    document.getElementById('bet-modal').classList.remove('active');
}

function refreshAllBets() {
    loadAllBets();
}

// Legacy support
async function loadPredictions() {
    loadAllBets();
}

async function loadEdgeAnalysis() {
    const tableBody = document.getElementById('edge-table-body');

    // Bail out if Edge Analysis section doesn't exist in current HTML
    if (!tableBody) {
        console.log('Edge Analysis section not present in DOM');
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/edge-analysis?sport=${currentSport}`);
        const data = await response.json();

        if (data.error) {
            tableBody.innerHTML = `<tr><td colspan="6">Error: ${data.error}</td></tr>`;
            return;
        }

        // Update summary elements (with null checks)
        const gamesAnalyzed = document.getElementById('games-analyzed');
        const gamesWithEdge = document.getElementById('games-with-edge');
        const avgEdge = document.getElementById('avg-edge');

        if (gamesAnalyzed) gamesAnalyzed.textContent = data.summary?.totalGames || 0;
        if (gamesWithEdge) gamesWithEdge.textContent = data.summary?.gamesWithEdge || 0;
        if (avgEdge) avgEdge.textContent = `${((data.summary?.avgEdge || 0) * 100).toFixed(1)}%`;

        // Update table
        if (!data.edges || data.edges.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="6">No edges found in current games</td></tr>';
            return;
        }

        tableBody.innerHTML = data.edges.map(edge => `
            <tr>
                <td>${edge.game}</td>
                <td><strong>${edge.team}</strong></td>
                <td>${formatOdds(edge.odds)}</td>
                <td>${(edge.modelProbability * 100).toFixed(1)}%</td>
                <td class="positive">+${(edge.ev * 100).toFixed(1)}%</td>
                <td><span class="badge win">BET</span></td>
            </tr>
        `).join('');

        // Render chart
        renderEdgeChart(data.edges);

    } catch (error) {
        if (tableBody) {
            tableBody.innerHTML = `<tr><td colspan="6">Failed to load: ${error.message}</td></tr>`;
        }
    }
}

async function loadPerformance() {
    try {
        const response = await fetch(`${API_BASE}/api/performance`);
        const data = await response.json();

        const perf = data.performance || {
            total_bets: 0,
            wins: 0,
            losses: 0,
            win_rate: 0,
            total_wagered: 0,
            total_returned: 0,
            profit: 0,
            roi: 0
        };

        document.getElementById('total-bets').textContent = perf.total_bets || 0;
        document.getElementById('win-rate').textContent = `${((perf.win_rate || 0) * 100).toFixed(1)}%`;
        document.getElementById('total-wagered').textContent = `$${(perf.total_wagered || 0).toLocaleString()}`;

        const profit = perf.profit || 0;
        const roi = perf.roi || 0;

        const profitEl = document.getElementById('profit');
        profitEl.textContent = `$${profit >= 0 ? '+' : ''}${profit.toLocaleString()}`;
        document.getElementById('profit-card').classList.toggle('positive', profit >= 0);
        document.getElementById('profit-card').classList.toggle('negative', profit < 0);

        const roiEl = document.getElementById('roi');
        roiEl.textContent = `${roi >= 0 ? '+' : ''}${(roi * 100).toFixed(1)}%`;
        document.getElementById('roi-card').classList.toggle('positive', roi >= 0);
        document.getElementById('roi-card').classList.toggle('negative', roi < 0);

        // Update Bankroll
        const bankroll = perf.current_bankroll || 100.00;
        document.getElementById('bankroll').textContent = `$${bankroll.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

        // Load performance history chart
        loadPerformanceChart();

        // Load recent bets table
        loadRecentBets();

    } catch (error) {
        console.error('Failed to load performance:', error);
    }
}

async function loadPerformanceChart() {
    const ctx = document.getElementById('profit-chart');
    if (!ctx) return;

    try {
        const response = await fetch(`${API_BASE}/api/performance-history`);
        const data = await response.json();

        if (!data.history || data.history.length === 0) {
            ctx.parentElement.innerHTML = '<div class="no-data">No performance history yet</div>';
            return;
        }

        const history = data.history;

        // Destroy previous chart if exists
        if (window.performanceChart) {
            window.performanceChart.destroy();
        }

        window.performanceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: history.map(h => h.date),
                datasets: [
                    {
                        label: 'Cumulative P/L ($)',
                        data: history.map(h => h.cumulative),
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 4,
                        pointHoverRadius: 6
                    },
                    {
                        label: 'Daily P/L ($)',
                        data: history.map(h => h.profit),
                        borderColor: '#6366f1',
                        backgroundColor: 'rgba(99, 102, 241, 0.1)',
                        fill: false,
                        tension: 0.3,
                        pointRadius: 4,
                        pointHoverRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#a0a8b8' }
                    },
                    tooltip: {
                        callbacks: {
                            afterLabel: (context) => {
                                const idx = context.dataIndex;
                                const h = history[idx];
                                return `Record: ${h.wins}W - ${h.losses}L`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#a0a8b8' },
                        grid: { color: '#2d3748' }
                    },
                    y: {
                        ticks: {
                            color: '#a0a8b8',
                            callback: value => '$' + value
                        },
                        grid: { color: '#2d3748' }
                    }
                }
            }
        });

        // Show demo notice if applicable
        if (data.isDemo) {
            const notice = document.createElement('div');
            notice.className = 'demo-notice';
            notice.innerHTML = '<small>📊 Demo data - actual history will appear after bets resolve</small>';
            ctx.parentElement.prepend(notice);
        }

    } catch (error) {
        console.error('Failed to load performance chart:', error);
    }
}

async function loadRecentBets() {
    const tableBody = document.getElementById('recent-bets-body');
    if (!tableBody) return;

    try {
        const statusFilter = document.getElementById('filter-recent-status')?.value || 'all';
        const response = await fetch(`${API_BASE}/api/recent-bets?status=${statusFilter}`);
        const data = await response.json();

        if (!data.bets || data.bets.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="7">No betting history found</td></tr>';
            return;
        }

        tableBody.innerHTML = data.bets.map(bet => {
            const resultClass = bet.result === 'WIN' ? 'win' : bet.result === 'LOSS' ? 'loss' : 'pending';
            const payoutClass = bet.payout && bet.payout.startsWith('+') ? 'positive' : 'negative';
            return `
                <tr>
                    <td>${bet.date}</td>
                    <td>${bet.sport}</td>
                    <td>${bet.game}</td>
                    <td><strong>${bet.team}</strong></td>
                    <td><span class="badge ${resultClass}">${bet.result}</span></td>
                    <td>${formatOdds(bet.odds)}</td>
                    <td class="${payoutClass}">${bet.payout || '-'}</td>
                </tr>
            `;
        }).join('');

    } catch (error) {
        console.error('Failed to load recent bets:', error);
        tableBody.innerHTML = '<tr><td colspan="7">Failed to load betting history</td></tr>';
    }
}

// ============== Rendering ==============

function renderPredictions(predictions) {
    const container = document.getElementById('predictions-container');

    container.innerHTML = predictions.map(pred => `
        <div class="prediction-card ${pred.hasEdge ? 'has-edge' : ''}">
            <div class="prediction-header">
                <div class="game-info">
                    <h3>${pred.awayAbbr} @ ${pred.homeAbbr}</h3>
                    <span class="game-time">${formatGameTime(pred.startTime)}</span>
                </div>
                <span class="confidence-badge ${getConfidenceClass(pred.confidence)}">
                    ${(pred.confidence * 100).toFixed(0)}% conf
                </span>
            </div>
            
            <div class="teams-comparison">
                <div class="team">
                    <div class="team-name">${pred.homeAbbr}</div>
                    <div class="team-prob">${(pred.homeProb * 100).toFixed(1)}%</div>
                    <div class="team-odds">${formatOdds(pred.homeOdds)}</div>
                </div>
                <div class="vs-divider">VS</div>
                <div class="team">
                    <div class="team-name">${pred.awayAbbr}</div>
                    <div class="team-prob">${(pred.awayProb * 100).toFixed(1)}%</div>
                    <div class="team-odds">${formatOdds(pred.awayOdds)}</div>
                </div>
            </div>
            
            <div class="prediction-stats">
                <div class="stat-row">
                    <span>Home Scoring (L5):</span>
                    <span>${pred.teamAnalysis?.home_pts_l5?.toFixed(1) || 'N/A'} PPG</span>
                </div>
                <div class="stat-row">
                    <span>Away Scoring (L5):</span>
                    <span>${pred.teamAnalysis?.away_pts_l5?.toFixed(1) || 'N/A'} PPG</span>
                </div>
                <div class="stat-row">
                    <span>Home EV:</span>
                    <span class="${pred.homeEv > 0 ? 'positive' : 'negative'}">
                        ${(pred.homeEv * 100).toFixed(1)}%
                    </span>
                </div>
                <div class="stat-row">
                    <span>Away EV:</span>
                    <span class="${pred.awayEv > 0 ? 'positive' : 'negative'}">
                        ${(pred.awayEv * 100).toFixed(1)}%
                    </span>
                </div>
            </div>
            
            <div class="recommendation ${pred.hasEdge ? 'bet' : 'pass'}">
                ${pred.hasEdge
            ? `>>> BET: ${pred.recommendedBet} (EV: +${(pred.recommendedBetEv * 100).toFixed(1)}%)`
            : 'No significant edge'}
            </div>
        </div>
    `).join('');
}

function renderEdgeChart(edges) {
    const ctx = document.getElementById('edge-chart');
    if (!ctx) return;

    // Destroy previous chart if exists
    if (window.edgeChart) {
        window.edgeChart.destroy();
    }

    window.edgeChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: edges.slice(0, 10).map(e => e.team),
            datasets: [{
                label: 'Expected Value (%)',
                data: edges.slice(0, 10).map(e => e.ev * 100),
                backgroundColor: edges.slice(0, 10).map(e =>
                    e.ev > 0.5 ? 'rgba(16, 185, 129, 0.8)' :
                        e.ev > 0.2 ? 'rgba(99, 102, 241, 0.8)' :
                            'rgba(139, 92, 246, 0.8)'
                ),
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    ticks: { color: '#a0a8b8' },
                    grid: { color: '#2d3748' }
                },
                y: {
                    ticks: {
                        color: '#a0a8b8',
                        callback: value => value + '%'
                    },
                    grid: { color: '#2d3748' }
                }
            }
        }
    });
}

// ============== Helpers ==============

function getConfidenceClass(confidence) {
    if (confidence >= 0.65) return 'confidence-high';
    if (confidence >= 0.55) return 'confidence-medium';
    return 'confidence-low';
}

function formatOdds(odds) {
    if (!odds && odds !== 0) return 'N/A';
    return odds > 0 ? `+${odds}` : `${odds}`;
}

function formatGameTime(isoString) {
    if (!isoString) return 'TBD';
    try {
        const date = new Date(isoString);
        return date.toLocaleTimeString('en-US', {
            hour: 'numeric',
            minute: '2-digit',
            hour12: true
        });
    } catch {
        return 'TBD';
    }
}

function refreshPredictions() {
    loadPredictions();
}

function changeSport(sport) {
    currentSport = sport;
    loadPredictions();
    loadEdgeAnalysis();
}

// ============== Bet Evaluator ==============

async function evaluateBet() {
    const sport = document.getElementById('eval-sport').value;
    const home = document.getElementById('eval-home').value.toUpperCase();
    const away = document.getElementById('eval-away').value.toUpperCase();
    const homeOdds = parseInt(document.getElementById('eval-home-odds').value);
    const awayOdds = parseInt(document.getElementById('eval-away-odds').value);
    const isHistorical = document.getElementById('eval-historical').checked;

    const resultDiv = document.getElementById('eval-result');
    resultDiv.innerHTML = '<div class="loading"></div>';

    try {
        const url = `${API_BASE}/api/predict/${home}/${away}?home_odds=${homeOdds}&away_odds=${awayOdds}&sport=${sport}&historical=${isHistorical}`;
        const response = await fetch(url);
        const data = await response.json();

        if (data.error) {
            resultDiv.innerHTML = `<div class="error-message">Error: ${data.error}</div>`;
            return;
        }

        const pred = data.prediction;
        const ba = pred.betting_analysis || {};
        const homeProb = pred.home_win_probability;
        const awayProb = pred.away_win_probability;
        const homeEv = ba.home_ev || 0;
        const awayEv = ba.away_ev || 0;

        const bestEv = Math.max(homeEv, awayEv);
        const bestTeam = homeEv > awayEv ? home : away;
        const bestOdds = homeEv > awayEv ? homeOdds : awayOdds;

        let recommendation = '';
        let recClass = '';
        if (bestEv > 0.05) {
            recommendation = `>>> STRONG BET: ${bestTeam} (${bestOdds > 0 ? '+' : ''}${bestOdds}) | EV: +${(bestEv * 100).toFixed(1)}%`;
            recClass = 'bet';
        } else if (bestEv > 0.02) {
            recommendation = `>>> SLIGHT EDGE: ${bestTeam} (${bestOdds > 0 ? '+' : ''}${bestOdds}) | EV: +${(bestEv * 100).toFixed(1)}%`;
            recClass = 'bet';
        } else if (bestEv > 0) {
            recommendation = `>>> MARGINAL: Consider passing (+${(bestEv * 100).toFixed(1)}% EV)`;
            recClass = 'pass';
        } else {
            recommendation = `>>> PASS: No positive EV available`;
            recClass = 'pass';
        }

        resultDiv.innerHTML = `
            <div class="eval-result-card">
                <div class="eval-matchup">${away} @ ${home}</div>
                <div class="eval-probs">
                    <div class="eval-team">
                        <div class="eval-team-name">${home}</div>
                        <div class="eval-team-prob">${(homeProb * 100).toFixed(1)}%</div>
                        <div class="eval-team-ev ${homeEv > 0 ? 'positive' : 'negative'}">
                            ${homeOdds > 0 ? '+' : ''}${homeOdds} → EV: ${(homeEv * 100).toFixed(1)}%
                        </div>
                    </div>
                    <div class="eval-team">
                        <div class="eval-team-name">${away}</div>
                        <div class="eval-team-prob">${(awayProb * 100).toFixed(1)}%</div>
                        <div class="eval-team-ev ${awayEv > 0 ? 'positive' : 'negative'}">
                            ${awayOdds > 0 ? '+' : ''}${awayOdds} → EV: ${(awayEv * 100).toFixed(1)}%
                        </div>
                    </div>
                </div>
                <div class="eval-recommendation ${recClass}">
                    ${recommendation}
                </div>
            </div>
        `;

    } catch (error) {
        resultDiv.innerHTML = `<div class="error-message">Failed to evaluate: ${error.message}</div>`;
    }
}

// ============== Team Dropdown Functions ==============

function updateTeamOptions() {
    const sport = document.getElementById('eval-sport').value;
    const homeSelect = document.getElementById('eval-home');
    const awaySelect = document.getElementById('eval-away');

    const teams = TEAMS[sport] || TEAMS.nba;

    homeSelect.innerHTML = teams.map(t => `<option value="${t.abbr}">${t.name} (${t.abbr})</option>`).join('');
    awaySelect.innerHTML = teams.map(t => `<option value="${t.abbr}">${t.name} (${t.abbr})</option>`).join('');

    // Set different defaults
    homeSelect.value = teams[0].abbr;
    awaySelect.value = teams[1].abbr;
}

// ============== Initialize ==============

document.addEventListener('DOMContentLoaded', () => {
    loadPredictions();

    // Initialize team dropdowns
    updateTeamOptions();

    // Auto-refresh every 5 minutes
    setInterval(() => {
        const activeTab = document.querySelector('.nav-btn.active')?.dataset.tab;
        if (activeTab === 'predictions') loadPredictions();
        if (activeTab === 'edge') loadEdgeAnalysis();
    }, 5 * 60 * 1000);
});

