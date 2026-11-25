migrate((app) => {
  const collection = app.findCollectionByNameOrId("users");

  collection.fields.add(new TextField({
    name: "name",
    required: true,
    min: 2,
    max: 80,
  }));
  collection.fields.add(new SelectField({
    name: "role",
    required: true,
    maxSelect: 1,
    values: ["admin", "user"],
  }));

  collection.listRule = '@request.auth.role = "admin"';
  collection.viewRule = '@request.auth.role = "admin"';
  collection.createRule = '@request.auth.role = "admin"';
  collection.updateRule = '@request.auth.role = "admin"';
  collection.deleteRule = '@request.auth.role = "admin"';
  collection.manageRule = '@request.auth.role = "admin"';
  collection.passwordAuth.enabled = true;
  collection.passwordAuth.identityFields = ["email"];

  return app.save(collection);
});
