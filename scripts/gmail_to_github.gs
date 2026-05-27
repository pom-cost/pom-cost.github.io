/**
 * gmail_to_github.gs — Google Apps Script
 *
 * Monitors Gmail for EnvLogger emails and pushes CSV attachments
 * directly to pom-cost/pom-cost.github.io/data/raw/ on GitHub.
 *
 * ── ONE-TIME SETUP ──────────────────────────────────────────────
 * 1. Go to https://script.google.com → New project
 * 2. Paste this entire file
 * 3. Project Settings (⚙) → Script Properties → Add property:
 *      Name:  GITHUB_TOKEN
 *      Value: <your GitHub Personal Access Token>
 *      (Token needs: Contents → Read and write, on pom-cost repo)
 * 4. In the editor, select function "installTrigger" → Run
 *    This installs an automatic hourly check. Do this only once.
 * 5. Authorize when prompted (allow Gmail + external requests)
 *
 * ── HOW IT WORKS ────────────────────────────────────────────────
 * Every hour, searches Gmail for emails with subject "EnvLogger"
 * that have CSV attachments and haven't been processed yet.
 * Each CSV is pushed to GitHub via the API. Processed emails get
 * a "envlogger-processed" label so they're never uploaded twice.
 * ────────────────────────────────────────────────────────────────
 */

var GITHUB_OWNER  = 'pom-cost';
var GITHUB_REPO   = 'pom-cost.github.io';
var GITHUB_BRANCH = 'redesign/beautiful-jekyll-style'; // update to 'main' after branch cleanup
var GITHUB_PATH   = 'data/raw';
var GMAIL_QUERY   = 'subject:EnvLogger has:attachment filename:csv';
var PROCESSED_LABEL = 'envlogger-processed';


/* ── Install hourly trigger (run once manually) ── */
function installTrigger() {
  // Remove existing triggers for this function to avoid duplicates
  ScriptApp.getProjectTriggers().forEach(function(t) {
    if (t.getHandlerFunction() === 'processNewEmails') {
      ScriptApp.deleteTrigger(t);
    }
  });
  ScriptApp.newTrigger('processNewEmails')
    .timeBased()
    .everyHours(1)
    .create();
  Logger.log('Trigger installed — will check Gmail every hour.');
}


/* ── Main function: find new emails and push attachments ── */
function processNewEmails() {
  var token = PropertiesService.getScriptProperties().getProperty('GITHUB_TOKEN');
  if (!token) {
    Logger.log('ERROR: GITHUB_TOKEN not set in Script Properties.');
    return;
  }

  // Get or create the Gmail label used to track processed emails
  var label = GmailApp.getUserLabelByName(PROCESSED_LABEL);
  if (!label) label = GmailApp.createLabel(PROCESSED_LABEL);

  // Find unprocessed EnvLogger emails
  var threads = GmailApp.search(GMAIL_QUERY + ' -label:' + PROCESSED_LABEL);
  Logger.log('New EnvLogger threads found: ' + threads.length);

  threads.forEach(function(thread) {
    var messages = thread.getMessages();
    messages.forEach(function(message) {
      var attachments = message.getAttachments();
      attachments.forEach(function(att) {
        var name = att.getName();
        var type = att.getContentType();
        if (name.endsWith('.csv') || type === 'text/csv' || type === 'application/octet-stream') {
          Logger.log('Processing attachment: ' + name);
          pushToGitHub(name, att.getDataAsString(), token);
        }
      });
    });
    // Mark thread so it won't be processed again
    thread.addLabel(label);
  });
}


/* ── Push a single file to GitHub via API ── */
function pushToGitHub(filename, content, token) {
  // Encode branch name for use in URL query params (handles slashes)
  var branchEncoded = encodeURIComponent(GITHUB_BRANCH);
  var fileUrl = 'https://api.github.com/repos/' + GITHUB_OWNER + '/' + GITHUB_REPO +
                '/contents/' + GITHUB_PATH + '/' + encodeURIComponent(filename);

  // Skip if file already exists on GitHub — never overwrite raw data
  try {
    var checkResp = UrlFetchApp.fetch(fileUrl + '?ref=' + branchEncoded, {
      headers: { Authorization: 'token ' + token },
      muteHttpExceptions: true
    });
    if (checkResp.getResponseCode() === 200) {
      Logger.log('⏭ File already exists, skipping: ' + filename);
      return;
    }
  } catch (e) {
    Logger.log('Could not check existing file: ' + e);
  }

  // Build the PUT payload
  var payload = {
    message: 'data: add raw sensor file ' + filename,
    content: Utilities.base64Encode(content),  // GitHub API requires base64
    branch: GITHUB_BRANCH
  };

  var resp = UrlFetchApp.fetch(fileUrl, {
    method: 'put',
    contentType: 'application/json',
    headers: { Authorization: 'token ' + token },
    payload: JSON.stringify(payload),
    muteHttpExceptions: true
  });

  var code = resp.getResponseCode();
  if (code === 201) {
    Logger.log('✓ New file pushed: ' + filename);
  } else if (code === 200) {
    Logger.log('✓ File updated: ' + filename);
  } else {
    Logger.log('✗ GitHub error ' + code + ' for ' + filename + ': ' + resp.getContentText());
  }
}


/* ── Test function: run manually to verify everything works ── */
function testScript() {
  var token = PropertiesService.getScriptProperties().getProperty('GITHUB_TOKEN');
  if (!token) {
    Logger.log('ERROR: GITHUB_TOKEN not set.');
    return;
  }
  Logger.log('Token found. Searching Gmail for EnvLogger emails...');
  var threads = GmailApp.search(GMAIL_QUERY);
  Logger.log('Total EnvLogger threads (all time): ' + threads.length);
  var unprocessed = GmailApp.search(GMAIL_QUERY + ' -label:' + PROCESSED_LABEL);
  Logger.log('Unprocessed: ' + unprocessed.length);
}
