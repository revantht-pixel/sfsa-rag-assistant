/*
 * SFSA Assistant loader -- paste this into MediaWiki:Common.js.
 *
 * Plain ES5, so it runs on MediaWiki 1.34 and later. It adds an "Ask SFSA
 * Assistant" button for logged-in members. The chat itself lives in an iframe
 * served from rag.sfsa.org, so nothing typed or shown in the chat is visible
 * to this page's other scripts (Google Analytics / Tag Manager).
 *
 * Deliberately small and stable: browsers cache wiki-side JavaScript for up
 * to 24 hours, so behaviour changes belong in the chat served by
 * rag.sfsa.org, not here. To point at a different server (for testing), change
 * CHAT_URL only.
 *
 * Privacy: the iframe is created only when a member first opens the assistant.
 * Merely viewing a wiki page never contacts rag.sfsa.org.
 *
 * Security: hiding the button from logged-out visitors is a convenience only.
 * rag.sfsa.org verifies the member's wiki login itself on every request.
 */
( function () {
	'use strict';

	var CHAT_URL = 'https://rag.sfsa.org/widget';

	if ( !window.mw || !mw.config || !mw.config.get( 'wgUserName' ) ) {
		return;
	}

	function init() {
		if ( document.getElementById( 'sfsa-assistant-launcher' ) ) {
			return;
		}

		var panel = null;
		var frame = null;
		var isOpen = false;

		var launcher = document.createElement( 'button' );
		launcher.id = 'sfsa-assistant-launcher';
		launcher.type = 'button';
		launcher.textContent = 'Ask SFSA Assistant';
		launcher.setAttribute( 'aria-expanded', 'false' );
		launcher.style.cssText = 'position:fixed;right:16px;bottom:16px;z-index:10000;padding:10px 16px;' +
			'border:0;border-radius:20px;background:#36454f;color:#fff;font:14px sans-serif;' +
			'cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,.3);';

		function buildPanel() {
			panel = document.createElement( 'div' );
			panel.id = 'sfsa-assistant-panel';
			panel.style.cssText = 'position:fixed;right:16px;bottom:64px;z-index:10000;width:380px;height:560px;' +
				'max-width:calc(100vw - 32px);max-height:calc(100vh - 96px);background:#fff;' +
				'border:1px solid #c9cdd3;border-radius:10px;box-shadow:0 4px 16px rgba(0,0,0,.25);' +
				'overflow:hidden;display:flex;flex-direction:column;';

			var bar = document.createElement( 'div' );
			bar.style.cssText = 'display:flex;justify-content:space-between;align-items:center;' +
				'padding:6px 10px;background:#36454f;color:#fff;font:13px sans-serif;';
			var title = document.createElement( 'span' );
			title.textContent = 'SFSA Assistant';
			var close = document.createElement( 'button' );
			close.type = 'button';
			close.textContent = '×';
			close.setAttribute( 'aria-label', 'Close the SFSA Assistant' );
			close.style.cssText = 'border:0;background:none;color:#fff;font-size:20px;line-height:1;cursor:pointer;';
			close.addEventListener( 'click', toggle );
			bar.appendChild( title );
			bar.appendChild( close );

			frame = document.createElement( 'iframe' );
			frame.title = 'SFSA Assistant';
			frame.setAttribute( 'sandbox', 'allow-scripts allow-same-origin allow-forms allow-popups allow-popups-to-escape-sandbox' );
			frame.setAttribute( 'referrerpolicy', 'no-referrer' );
			frame.setAttribute( 'allow', '' );
			frame.style.cssText = 'flex:1;width:100%;border:0;';
			frame.src = CHAT_URL;

			panel.appendChild( bar );
			panel.appendChild( frame );
			document.body.appendChild( panel );
		}

		function toggle() {
			if ( !panel ) {
				buildPanel();
			}
			isOpen = !isOpen;
			panel.style.display = isOpen ? 'flex' : 'none';
			launcher.setAttribute( 'aria-expanded', isOpen ? 'true' : 'false' );
		}

		launcher.addEventListener( 'click', toggle );
		document.addEventListener( 'keydown', function ( e ) {
			if ( isOpen && ( e.key === 'Escape' || e.keyCode === 27 ) ) {
				toggle();
			}
		} );
		document.body.appendChild( launcher );
	}

	if ( document.body ) {
		init();
	} else {
		document.addEventListener( 'DOMContentLoaded', init );
	}
}() );
