-- SQLite
SELECT 
    SUBSTR(
        string_value,
        INSTR(string_value, '/Users/andriikozin/prj/ms/wiki/IdentityWiki/Services/') + LENGTH('/Users/andriikozin/prj/ms/wiki/IdentityWiki/Services/'),
        INSTR(SUBSTR(string_value, INSTR(string_value, '/Users/andriikozin/prj/ms/wiki/IdentityWiki/Services/') + LENGTH('/Users/andriikozin/prj/ms/wiki/IdentityWiki/Services/')), '/') - 1
    ) AS service_name,
    COUNT(DISTINCT string_value) AS count
FROM embedding_metadata
WHERE `key` = 'source'
AND string_value LIKE '/Users/andriikozin/prj/ms/wiki/IdentityWiki/Services/%'
GROUP BY service_name

UNION ALL

SELECT 
    'total' AS service_name,
    COUNT(DISTINCT string_value) AS count
FROM embedding_metadata
WHERE `key` = 'source'
AND string_value LIKE '/Users/andriikozin/prj/ms/wiki/IdentityWiki/Services/%'

ORDER BY service_name;